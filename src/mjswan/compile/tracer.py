"""Trace real mjlab MDP term bodies to ONNX (ADR 0005, Phase 1).

The ONNX rewrite (ADR 0005) exports each Observation/Termination/Event/Command
*term body* to an ONNX graph at build time and runs it with ONNX Runtime Web in
the browser, instead of reimplementing the math as a native DSL primitive
(ADR 0003). This module implements the build-time tracing for **value-returning
terms** (observations and non-native terminations) — the core mechanism the rest
of the rewrite builds on.

The key idea: an mjlab term is written as ``func(env, **params)`` and reads a few
fields off ``env`` (e.g. ``env.scene["robot"].data.joint_pos``). To trace it with
``torch.onnx.export`` we:

1. **Discover** which ``env.scene[name].data.<field>`` tensors the function reads,
   by running it once against a recording proxy that wraps the *real* env.
2. **Classify** each accessed field as either time-varying simulation state (an
   ONNX graph input) or a constant (baked into the graph at trace time). The
   split is a small field-name registry — far cheaper than reimplementing the
   term's math, and it is the only knowledge this layer carries.
3. **Wrap** the function in an ``nn.Module`` whose ``forward`` takes only the
   dynamic tensors, serves the constants from registered buffers, and calls the
   real ``func``. ``torch.onnx.export`` then records the actual torch ops.

Statically-resolved indices (``asset_cfg.joint_ids`` etc.) are ordinary Python
values closed over by the function, so they bake into the graph as constants for
free — exactly as ADR 0005 §Consequences requires.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Field classification — the only mjlab knowledge this layer carries.
#
# A field read off ``Entity.data`` is a **graph input** if it is time-varying
# simulation state, otherwise it is **baked as a constant** at trace time. Keep
# this in sync with the browser runtime's state-collection (`collectRawState`,
# ADR 0005 §6) as new mjlab data fields are exercised.
# ---------------------------------------------------------------------------

_DYNAMIC_DATA_FIELDS: frozenset[str] = frozenset(
    {
        "joint_pos",
        "joint_pos_biased",
        "joint_vel",
        "root_link_pos_w",
        "root_link_quat_w",
        "root_link_lin_vel_b",
        "root_link_ang_vel_b",
        "root_ang_vel_b",
        "projected_gravity_b",
    }
)

# A slot key identifies one tensor read off the env: (entity_name, data_field).
SlotKey = tuple[str, str]


# ---------------------------------------------------------------------------
# Recording proxy — discovers which env fields a term reads.
# ---------------------------------------------------------------------------


class _RecordingData:
    """Wraps a real ``Entity.data``, logging every field access."""

    def __init__(self, real: Any, entity: str, log: list[tuple[SlotKey, Any]]):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._real, name)
        self._log.append(((self._entity, name), value))
        return value


class _RecordingEntity:
    def __init__(self, real: Any, name: str, log: list[tuple[SlotKey, Any]]):
        self._real = real
        self.data = _RecordingData(real.data, name, log)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _RecordingScene:
    def __init__(self, real: Any, log: list[tuple[SlotKey, Any]]):
        self._real = real
        self._log = log

    def __getitem__(self, name: str) -> _RecordingEntity:
        return _RecordingEntity(self._real[name], name, self._log)


class _RecordingEnv:
    """Proxy env that records which ``scene[name].data.<field>`` a term reads."""

    def __init__(self, real: Any):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_log", [])
        object.__setattr__(self, "scene", _RecordingScene(real.scene, self._log))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Replay proxy — serves recorded slots to the term during tracing.
# ---------------------------------------------------------------------------


class _ReplayData:
    def __init__(self, entity: str, slots: dict[SlotKey, torch.Tensor]):
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_slots", slots)

    def __getattr__(self, name: str) -> torch.Tensor:
        key = (self._entity, name)
        try:
            return self._slots[key]
        except KeyError:
            raise AttributeError(
                f"Term read undeclared slot {key!r} during tracing. This field "
                "was not seen in the discovery pass — the term's control flow is "
                "input-dependent, which is not traceable (ADR 0005 §Consequences)."
            ) from None


class _ReplayEntity:
    def __init__(self, entity: str, slots: dict[SlotKey, torch.Tensor]):
        self.data = _ReplayData(entity, slots)


class _ReplayScene:
    def __init__(self, slots: dict[SlotKey, torch.Tensor]):
        self._slots = slots

    def __getitem__(self, name: str) -> _ReplayEntity:
        return _ReplayEntity(name, self._slots)


class _ReplayEnv:
    def __init__(self, slots: dict[SlotKey, torch.Tensor]):
        self.scene = _ReplayScene(slots)


class _TermModule(nn.Module):
    """Wraps ``func(env, **params)`` so ``forward`` takes only dynamic tensors.

    Constant slots (defaults, static tensors) are registered as buffers and
    served back to the term; dynamic slots arrive as ``forward`` arguments in the
    order given by ``dynamic_keys``.
    """

    def __init__(
        self,
        func: Callable[..., torch.Tensor],
        params: dict[str, Any],
        dynamic_keys: list[SlotKey],
        constants: dict[SlotKey, torch.Tensor],
    ):
        super().__init__()
        self._func = func
        self._params = params
        self._dynamic_keys = dynamic_keys
        self._const_buffers: dict[SlotKey, str] = {}
        for i, (key, value) in enumerate(constants.items()):
            buffer_name = f"_const_{i}"
            self.register_buffer(buffer_name, value.detach().clone())
            self._const_buffers[key] = buffer_name

    def forward(self, *dynamic: torch.Tensor) -> torch.Tensor:
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, dynamic))
        for key, buffer_name in self._const_buffers.items():
            slots[key] = getattr(self, buffer_name)
        return self._func(_ReplayEnv(slots), **self._params)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class TermExport:
    """The result of tracing one term body to ONNX."""

    name: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    """Dynamic input slots, in ONNX graph input order — ``(entity, field)`` each."""
    input_names: list[str]
    output_name: str
    reference_output: torch.Tensor
    """The term's output on the discovery step (for a trace-time sanity check)."""
    constant_slots: list[SlotKey] = field(default_factory=list)

    @property
    def is_dynamic_only(self) -> bool:
        return len(self.input_slots) > 0


def _slot_input_name(key: SlotKey) -> str:
    entity, field_name = key
    return f"{entity}__{field_name}"


def trace_term(
    func: Callable[..., torch.Tensor],
    params: dict[str, Any],
    env: Any,
    *,
    name: str,
    opset: int = 17,
) -> TermExport:
    """Trace a value-returning mjlab term body to ONNX against a live ``env``.

    Args:
        func: The mjlab term function, ``func(env, **params) -> Tensor``.
        params: Resolved params from the env's manager (``asset_cfg`` already
            resolved to static indices).
        env: A constructed mjlab ``ManagerBasedRlEnv`` (post-reset).
        name: Term name, used for input/output naming and diagnostics.
        opset: ONNX opset version.

    Returns:
        A :class:`TermExport` with the serialized ONNX graph and its input slots.

    Raises:
        ValueError: if the term reads no dynamic simulation state (it should then
            be handled as a native term, e.g. ``time_out``), or if tracing fails.
    """
    # 1. Discovery: run once against the recording env.
    recorder = _RecordingEnv(env)
    recorded = func(recorder, **params)
    if not isinstance(recorded, torch.Tensor):
        raise ValueError(
            f"Term {name!r} returned {type(recorded).__name__}, not a Tensor; "
            "only value-returning terms are traced here."
        )

    # 2. Classify accessed slots into dynamic inputs vs baked constants,
    #    de-duplicated and deterministically ordered.
    dynamic: dict[SlotKey, torch.Tensor] = {}
    constants: dict[SlotKey, torch.Tensor] = {}
    for key, value in recorder._log:  # noqa: SLF001 — internal proxy
        if not isinstance(value, torch.Tensor):
            continue  # non-tensor attribute access, not a graph slot
        _, field_name = key
        bucket = dynamic if field_name in _DYNAMIC_DATA_FIELDS else constants
        bucket.setdefault(key, value)

    if not dynamic:
        raise ValueError(
            f"Term {name!r} reads no time-varying state; handle it as a native "
            "term (e.g. time_out), not an ONNX graph (ADR 0005)."
        )

    dynamic_keys = sorted(dynamic)
    input_names = [_slot_input_name(k) for k in dynamic_keys]
    example_inputs = tuple(dynamic[k] for k in dynamic_keys)

    # 3. Trace to ONNX.
    module = _TermModule(func, params, dynamic_keys, constants).eval()
    output_name = "value"
    buffer = io.BytesIO()
    # The legacy TorchScript tracer records the concrete tensor ops the term runs
    # against the replay proxy, which is exactly the graph we want. The newer
    # torch.export path traces Python control flow and does not cope with the
    # proxy indirection; ``dynamo=False`` is deliberate here.
    with torch.no_grad():
        torch.onnx.export(
            module,
            example_inputs,
            buffer,
            input_names=input_names,
            output_names=[output_name],
            dynamic_axes={n: {0: "batch"} for n in [*input_names, output_name]},
            opset_version=opset,
            dynamo=False,
        )

    return TermExport(
        name=name,
        onnx_bytes=buffer.getvalue(),
        input_slots=dynamic_keys,
        input_names=input_names,
        output_name=output_name,
        reference_output=recorded.detach(),
        constant_slots=sorted(constants),
    )


def read_slot(env: Any, key: SlotKey) -> torch.Tensor:
    """Read the current value of an input slot ``(entity, field)`` from ``env``."""
    entity, field_name = key
    return getattr(env.scene[entity].data, field_name)
