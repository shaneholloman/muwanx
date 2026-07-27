"""Build-time ONNX serialization helpers (ADR 0005).

Bridges the config-side dataclasses (``ObservationGroupCfg``,
``TerminationTermCfg``, ``EventTermCfg``, ``CommandTermConfig``) to the
``mjswan.compile`` tracer: traces each plain-callable term body against the
scene's live mjlab env, writes the resulting ``.onnx`` bytes under the scene's
output directory, and returns the manifest-shaped JSON entry the runtime
consumes. Legacy ``*Binding``-typed terms (``ts_src`` / built-in named classes)
still serialize via each cfg's own ``to_dict()`` — only the *representation* of
author-authored term bodies changes, not the wire contract for the rest.

Called from :mod:`mjswan.builder`, once per scene, after that scene's live
``mjlab_env`` and output directory (``scene_dir``) are both known.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .command import CommandTermConfig
from .envs.mdp.events import EventBinding
from .envs.mdp.observations import ObservationBinding
from .envs.mdp.terminations import TerminationBinding

if TYPE_CHECKING:
    from .managers.event_manager import EventTermCfg
    from .managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from .managers.termination_manager import TerminationTermCfg


def _onnx_ref(kind: str, name: str) -> str:
    """Bundle-relative path for a traced term's ``.onnx`` file."""
    return f"{kind}/{name}.onnx"


def _write_onnx(out_dir: Path, ref: str, onnx_bytes: bytes) -> None:
    path = out_dir / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_bytes)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def _native_observation_entry(
    name: str, func: Any, params: dict[str, Any]
) -> dict[str, Any] | None:
    """Classify a known non-``entity.data`` observation func into a native marker.

    Two mjlab functions are legitimately native by design: ``last_action``
    reads ``env.action_manager.action`` and ``generated_commands`` reads
    ``env.command_manager.get_command(...)`` — both env-level, not
    ``entity.data``, so the tracer's recording proxy (which only wraps
    ``env.scene``) never sees them, and ``trace_term`` would raise ``ValueError``
    (no dynamic state). Checked *before* attempting to trace (rather than
    catching that ``ValueError``): a scene without the named command traced
    (e.g. a demo pairing ``generated_commands`` with a purely-native
    ``UiCommand``, not an ``OnnxCommand``) would raise from mjlab's own
    ``assert command is not None`` during the discovery call itself, not a
    clean ``ValueError``. Both are already computed natively every frame by
    the TS orchestrator (the policy's previous output, and the named
    command's current value), so no ONNX graph is needed — the observation
    pipeline substitutes the live value directly. Returns ``None`` if *func*
    isn't one of these two — the caller should attempt tracing as normal.
    """
    func_name = getattr(func, "__name__", None)
    if func_name == "last_action":
        entry: dict[str, Any] = {"name": name, "native": "prev_action"}
        action_name = params.get("action_name")
        if action_name is not None:
            entry["action_name"] = action_name
        return entry
    if func_name == "generated_commands":
        return {
            "name": name,
            "native": "command",
            "command_name": params["command_name"],
        }
    return None


def _apply_observation_pipeline(
    entry: dict[str, Any],
    term_cfg: ObservationTermCfg,
    group_history_length: int | None,
) -> dict[str, Any]:
    """Add scale/clip/history metadata shared by every entry shape (traced,
    native, or baked-constant) — mirrors mjlab's compute -> scale -> history
    pipeline order (noise/delay are training-only, dropped, ADR 0005)."""
    if term_cfg.scale is not None:
        entry["scale"] = (
            list(term_cfg.scale)
            if isinstance(term_cfg.scale, tuple)
            else term_cfg.scale
        )
    if term_cfg.clip is not None:
        entry["clip"] = list(term_cfg.clip)
    history = (
        group_history_length
        if (group_history_length or 0) > 0
        else term_cfg.history_length
    )
    if history:
        entry["history_length"] = history
        if term_cfg.history_interleaved:
            entry["history_interleaved"] = True
    return entry


def serialize_observation_term(
    name: str,
    term_cfg: ObservationTermCfg,
    env: Any,
    out_dir: Path,
    group_history_length: int | None,
) -> dict[str, Any] | None:
    """Serialize one observation term. Returns ``None`` for a dropped/unsupported term."""
    from .compile import trace_term

    func = term_cfg.func
    if isinstance(func, ObservationBinding):
        if func.unsupported_reason is not None:
            return None
        return term_cfg.to_dict()

    native_entry = _native_observation_entry(name, func, term_cfg.params)
    if native_entry is not None:
        return _apply_observation_pipeline(native_entry, term_cfg, group_history_length)

    try:
        export = trace_term(func, term_cfg.params, env, name=name)
    except ValueError:
        # Not one of the two known native shapes above, and no
        # `entity.data.<field>` reads at all -- genuinely constant-valued
        # (e.g. a fixed-size placeholder/padding term with no env
        # dependency). Bake the value directly from a real call against the
        # live env; no graph needed. Re-raises if this isn't even a Tensor
        # (fail loud on anything genuinely unexpected).
        import torch

        value = func(env, **term_cfg.params)
        if not isinstance(value, torch.Tensor):
            raise
        entry = {
            "name": name,
            "native": "constant",
            "value": value.detach().flatten().tolist(),
        }
        return _apply_observation_pipeline(entry, term_cfg, group_history_length)
    ref = _onnx_ref("obs", name)
    _write_onnx(out_dir, ref, export.onnx_bytes)

    entry = {"name": name, "onnx": ref}
    return _apply_observation_pipeline(entry, term_cfg, group_history_length)


def serialize_observation_group(
    group: ObservationGroupCfg, env: Any, out_dir: Path
) -> list[dict[str, Any]]:
    """Serialize every term in an observation group to a JSON-ready list."""
    result = []
    for name, term_cfg in group.terms.items():
        entry = serialize_observation_term(
            name, term_cfg, env, out_dir, group.history_length
        )
        if entry is not None:
            result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def serialize_termination(
    name: str, term_cfg: TerminationTermCfg, env: Any, out_dir: Path
) -> dict[str, Any] | None:
    """Serialize one termination term. Returns ``None`` for an unsupported legacy term."""
    from .compile import trace_term

    func = term_cfg.func
    if isinstance(func, TerminationBinding):
        if func.unsupported_reason is not None:
            return None
        return term_cfg.to_dict()

    try:
        export = trace_term(func, term_cfg.params, env, name=name)
    except ValueError:
        # No time-varying state read (e.g. mjlab's `time_out`, which compares
        # env-level step counters, not entity data) -- this is the one
        # legitimately-native termination shape ADR 0005 §2 documents.
        entry: dict[str, Any] = {
            "name": name,
            "native": "elapsed_s >= episode_length_s",
        }
        if term_cfg.time_out:
            entry["time_out"] = True
        return entry

    ref = _onnx_ref("term", name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    entry = {"name": name, "onnx": ref}
    if term_cfg.time_out:
        entry["time_out"] = True
    return entry


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def serialize_event(
    name: str, term_cfg: EventTermCfg, env: Any, out_dir: Path
) -> dict[str, Any] | None:
    """Serialize one event term (any mode). Returns ``None`` if genuinely nothing to emit."""
    from .compile import trace_event_term
    from .compile.tracer import slot_to_json

    func = term_cfg.func
    if isinstance(func, EventBinding):
        return term_cfg.to_dict()

    try:
        export = trace_event_term(
            func, term_cfg.params, env, name=name, mode=term_cfg.mode
        )
    except ValueError as exc:
        # A model-field write (e.g. geom_friction/encoder_bias/body_com_offset
        # domain randomization) isn't captured by the joint/root entity_write
        # tracer yet -- known gap (companion brief §4's separate track), not a
        # build failure: surface it so a task author can see what was skipped.
        return {"name": name, "mode": term_cfg.mode, "native": True, "reason": str(exc)}

    ref = _onnx_ref("event", name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    entry: dict[str, Any] = {
        "name": name,
        "mode": term_cfg.mode,
        "onnx": ref,
        "rand_dim": export.rand_dim,
        "input_slots": [slot_to_json(k) for k in export.input_slots],
        "write_targets": export.write_targets,
    }
    if term_cfg.mode == "interval":
        entry["interval_range_s"] = (
            list(term_cfg.interval_range_s) if term_cfg.interval_range_s else None
        )
        entry["is_global_time"] = term_cfg.is_global_time
    if term_cfg.mode == "reset" and term_cfg.min_step_count_between_reset:
        entry["min_step_count_between_reset"] = term_cfg.min_step_count_between_reset
    return entry


def serialize_events(
    events: dict[str, EventTermCfg] | None, env: Any, out_dir: Path
) -> list[dict[str, Any]] | None:
    """Serialize a scene's events dict to the JSON list ``config.json`` carries."""
    if not events:
        return None
    result = []
    for name, term_cfg in events.items():
        entry = serialize_event(name, term_cfg, env, out_dir)
        if entry is not None:
            result.append(entry)
    return result or None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def serialize_command(
    name: str, cmd_cfg: CommandTermConfig, env: Any, out_dir: Path
) -> dict[str, Any]:
    """Serialize one command term, resolving a pending ONNX trace if needed."""
    if cmd_cfg.pending_trace is None:
        return cmd_cfg.to_dict()

    from .compile import trace_command_term
    from .compile.serialize import write_command_artifact

    pending = cmd_cfg.pending_trace
    term = pending.mjlab_cfg.build(env)
    if pending.trace_override is not None:
        pending.trace_override(term)

    export = trace_command_term(
        term,
        pending.state_fields,
        name=name,
        command_field=pending.command_field,
    )
    return write_command_artifact(
        export,
        out_dir,
        resampling_time_range=getattr(pending.mjlab_cfg, "resampling_time_range", None),
        debug_vis=bool(getattr(pending.mjlab_cfg, "debug_vis", False)),
        ui=pending.ui,
        viz=pending.viz,
    )
