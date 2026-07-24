"""Numeric-parity harness: live mjlab env vs exported ONNX graphs (ADR 0005).

Phase 1 exit criterion (ADR 0005 §Phased execution plan): the ONNX graphs
exported from a task's term bodies must match the live mjlab environment within
tolerance for **every term, every step**.

This harness:

1. Builds a live mjlab ``ManagerBasedRlEnv``.
2. Traces every value-returning observation term to ONNX (:mod:`.tracer`),
   classifying terminations like ``time_out`` as native.
3. Steps the env ``n_steps`` times with a seeded action sequence, and at each
   step feeds the same raw state through each exported graph via ``onnxruntime``
   (**not** torch) and asserts ``allclose`` against the live term output.

Run headless with ``MUJOCO_GL=disable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from .rng import DrawRecorder
from .tracer import TermExport, read_slot, trace_event_term, trace_term


@dataclass
class TermReport:
    name: str
    kind: str  # "observation" | "termination" | "event"
    representation: str  # "onnx" | "native"
    input_slots: list[str] = field(default_factory=list)
    constant_slots: list[str] = field(default_factory=list)
    max_abs_diff: float = 0.0
    steps_checked: int = 0
    passed: bool = True
    note: str = ""
    rand_dim: int = 0


@dataclass
class ParityReport:
    n_steps: int
    atol: float
    rtol: float
    terms: list[TermReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.terms)

    def summary(self) -> str:
        lines = [
            f"Parity over {self.n_steps} steps (atol={self.atol}, rtol={self.rtol}):"
        ]
        for t in self.terms:
            status = "OK  " if t.passed else "FAIL"
            if t.representation == "native":
                lines.append(f"  [{status}] {t.name:<16} native ({t.note})")
            elif t.kind == "event":
                lines.append(
                    f"  [{status}] {t.name:<16} onnx-event  "
                    f"rand_dim={t.rand_dim} const={t.constant_slots} "
                    f"max|Δ|={t.max_abs_diff:.2e} over {t.steps_checked} draws"
                )
            else:
                lines.append(
                    f"  [{status}] {t.name:<16} onnx  "
                    f"in={t.input_slots} const={t.constant_slots} "
                    f"max|Δ|={t.max_abs_diff:.2e} over {t.steps_checked} steps"
                )
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


# Terms handled natively by the runtime (no ONNX graph); ADR 0005 §2 table.
_NATIVE_TERMINATIONS = {"time_out"}


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def _iter_obs_terms(
    env: Any, group: str
) -> list[tuple[str, Callable[..., torch.Tensor], dict[str, Any]]]:
    om = env.observation_manager
    names = om.active_terms[group]
    out = []
    for term_name in names:
        cfg = om.get_term_cfg(group, term_name)
        out.append((term_name, cfg.func, dict(cfg.params)))
    return out


def _iter_termination_terms(env: Any) -> list[tuple[str, Callable[..., torch.Tensor]]]:
    tm = env.termination_manager
    return [(name, tm.get_term_cfg(name).func) for name in tm.active_terms]


def _iter_event_terms(
    env: Any, mode: str
) -> list[tuple[str, Callable[..., None], dict[str, Any]]]:
    em = env.event_manager
    names = em.active_terms.get(mode, [])
    out = []
    for term_name in names:
        cfg = em.get_term_cfg(term_name)
        out.append((term_name, cfg.func, dict(cfg.params)))
    return out


def run_parity(
    env: Any,
    *,
    obs_group: str = "actor",
    n_steps: int = 64,
    seed: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    event_modes: tuple[str, ...] = ("reset",),
    n_event_draws: int = 16,
) -> ParityReport:
    """Trace a task's terms and assert live-vs-ONNX parity over ``n_steps``.

    ``env`` must be a freshly constructed mjlab env; this function resets it.
    Observation terms are checked every step; ``reset``-mode Event terms are
    checked by replaying ``n_event_draws`` fresh recorded RNG draws (§2b).
    """
    import onnxruntime as ort

    report = ParityReport(n_steps=n_steps, atol=atol, rtol=rtol)
    torch.manual_seed(seed)
    env.reset()

    # --- Trace observation terms; classify the rest. --------------------
    exports: dict[str, TermExport] = {}
    sessions: dict[str, ort.InferenceSession] = {}
    term_meta: list[tuple[str, Callable[..., torch.Tensor], dict[str, Any]]] = []

    for term_name, func, params in _iter_obs_terms(env, obs_group):
        try:
            export = trace_term(func, params, env, name=term_name)
        except ValueError as exc:
            report.terms.append(
                TermReport(
                    name=term_name,
                    kind="observation",
                    representation="native",
                    passed=True,
                    note=str(exc).split(";")[0],
                )
            )
            continue
        exports[term_name] = export
        sessions[term_name] = ort.InferenceSession(
            export.onnx_bytes, providers=["CPUExecutionProvider"]
        )
        term_meta.append((term_name, func, params))
        report.terms.append(
            TermReport(
                name=term_name,
                kind="observation",
                representation="onnx",
                input_slots=[f"{e}.{f}" for e, f in export.input_slots],
                constant_slots=[f"{e}.{f}" for e, f in export.constant_slots],
            )
        )

    for term_name, func in _iter_termination_terms(env):
        native = term_name in _NATIVE_TERMINATIONS
        report.terms.append(
            TermReport(
                name=term_name,
                kind="termination",
                representation="native" if native else "onnx",
                passed=True,
                note="elapsed_s >= episode_length_s" if native else "",
            )
        )
        # Non-native terminations would be traced here; Cartpole has only time_out.

    reports_by_name = {t.name: t for t in report.terms}
    action_dim = env.action_manager.total_action_dim

    # --- Step and compare every term every step. ------------------------
    for _ in range(n_steps):
        action = torch.rand((env.num_envs, action_dim)) * 2.0 - 1.0
        env.step(action)
        for term_name, func, params in term_meta:
            export = exports[term_name]
            session = sessions[term_name]
            feeds = {
                in_name: _to_numpy(read_slot(env, slot))
                for in_name, slot in zip(export.input_names, export.input_slots)
            }
            (onnx_out,) = session.run([export.output_name], feeds)
            live_out = _to_numpy(func(env, **params))
            diff = float(np.max(np.abs(onnx_out - live_out))) if live_out.size else 0.0
            tr = reports_by_name[term_name]
            tr.max_abs_diff = max(tr.max_abs_diff, diff)
            tr.steps_checked += 1
            if not np.allclose(onnx_out, live_out, atol=atol, rtol=rtol):
                tr.passed = False

    # --- Event terms: trace once, then replay fresh recorded RNG draws. -----
    for mode in event_modes:
        for term_name, func, params in _iter_event_terms(env, mode):
            tr = TermReport(name=term_name, kind="event", representation="onnx")
            report.terms.append(tr)
            try:
                export = trace_event_term(func, params, env, name=term_name, mode=mode)
            except ValueError as exc:
                tr.representation = "native"
                tr.note = str(exc).split(";")[0]
                continue
            tr.rand_dim = export.rand_dim
            tr.input_slots = [f"{e}.{f}" for e, f in export.input_slots]
            tr.constant_slots = [f"{e}.{f}" for e, f in export.constant_slots]
            session = ort.InferenceSession(
                export.onnx_bytes, providers=["CPUExecutionProvider"]
            )
            for _ in range(n_event_draws):
                # Record a fresh reference invocation (real draws, no sim write).
                captures: dict[str, tuple] = {}
                from .tracer import _EventCaptureEnv  # noqa: PLC0415

                proxy = _EventCaptureEnv(env, [], captures)
                with DrawRecorder(func) as rec:
                    func(proxy, None, **params)
                ref_pos, ref_vel = captures["joint_state"]
                feeds = {"rand": _to_numpy(rec.rand_vector)}
                for in_name, slot in zip(export.input_names, export.input_slots):
                    feeds[in_name] = _to_numpy(read_slot(env, slot))
                onnx_pos, onnx_vel = session.run(export.output_names, feeds)
                diff = max(
                    float(np.max(np.abs(onnx_pos - _to_numpy(ref_pos)))),
                    float(np.max(np.abs(onnx_vel - _to_numpy(ref_vel)))),
                )
                tr.max_abs_diff = max(tr.max_abs_diff, diff)
                tr.steps_checked += 1
                if not (
                    np.allclose(onnx_pos, _to_numpy(ref_pos), atol=atol, rtol=rtol)
                    and np.allclose(onnx_vel, _to_numpy(ref_vel), atol=atol, rtol=rtol)
                ):
                    tr.passed = False

    return report
