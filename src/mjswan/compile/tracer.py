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
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from .rng import DrawRecorder, ReplayRng

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
        "root_link_vel_w",
        "root_link_lin_vel_w",
        "root_link_ang_vel_w",
        "projected_gravity_b",
        "heading_w",
    }
)

# A slot key identifies one tensor read off the env, as ``(namespace, name)``:
#   (entity_name, data_field)  -> env.scene[entity].data.<field>
#   (_SENSOR_NS, sensor_name)  -> env.scene[sensor].data  (a whole BuiltinSensor read)
SlotKey = tuple[str, str]

_SENSOR_NS = "__sensor__"
"""Namespace marking a :class:`SlotKey` as a sensor read rather than entity data.

mjlab's scene indexes sensors and entities in one ``scene[name]`` namespace, so the
slot key needs its own marker to stay unambiguous. Not a legal mjlab entity name in
practice (mjlab names come from MJCF bodies/sites)."""


def _sensor_proxy(real: Any, get_data: Callable[[], Any]) -> Any:
    """A stand-in for a live mjlab ``Sensor`` whose ``.data`` comes from ``get_data``.

    Sensor-reading terms assert on the concrete class (``builtin_sensor`` does
    ``assert isinstance(sensor, BuiltinSensor)``), so a duck-typed proxy like the
    entity ones is rejected outright. Subclassing the *real* sensor's own class and
    sharing its ``__dict__`` keeps that check true — and keeps every other sensor
    attribute (``cfg``, ``num_frames``, …) working — while overriding only ``data``.
    """
    cls = type(real)
    proxy_cls = type(
        f"_Proxy{cls.__name__}", (cls,), {"data": property(lambda _self: get_data())}
    )
    proxy = object.__new__(proxy_cls)
    proxy.__dict__ = real.__dict__
    return proxy


def _is_sensor(scene: Any, name: str) -> bool:
    """Whether ``scene[name]`` resolves to a sensor rather than an entity.

    Asks the real scene's own ``sensors`` mapping rather than sniffing the object,
    so it matches mjlab's own ``Scene.__getitem__`` resolution order exactly.
    """
    sensors = getattr(scene, "sensors", None)
    return bool(sensors) and name in sensors


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
    def __init__(
        self,
        real: Any,
        log: list[tuple[SlotKey, Any]],
        sensors: dict[str, Any],
    ):
        self._real = real
        self._log = log
        self._sensors = sensors

    def __getitem__(self, name: str) -> Any:
        real = self._real[name]
        if _is_sensor(self._real, name):
            # Keep the real sensor so the replay pass can subclass its class.
            self._sensors[name] = real
            return _sensor_proxy(real, lambda: self._read_sensor(name, real))
        return _RecordingEntity(real, name, self._log)

    def _read_sensor(self, name: str, real: Any) -> Any:
        value = real.data
        self._log.append(((_SENSOR_NS, name), value))
        return value


class _RecordingEnv:
    """Proxy env recording which ``scene[...]`` reads (entity data / sensors) a term makes."""

    def __init__(self, real: Any):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_log", [])
        object.__setattr__(self, "_sensors", {})
        object.__setattr__(
            self, "scene", _RecordingScene(real.scene, self._log, self._sensors)
        )

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
    def __init__(
        self,
        slots: dict[SlotKey, torch.Tensor],
        sensors: dict[str, Any] | None = None,
    ):
        self._slots = slots
        self._sensors = sensors or {}

    def __getitem__(self, name: str) -> Any:
        real = self._sensors.get(name)
        if real is not None:
            return _sensor_proxy(real, lambda: self._slots[(_SENSOR_NS, name)])
        return _ReplayEntity(name, self._slots)


class _ReplayEnv:
    def __init__(
        self,
        slots: dict[SlotKey, torch.Tensor],
        sensors: dict[str, Any] | None = None,
    ):
        self.scene = _ReplayScene(slots, sensors)


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
        sensors: dict[str, Any] | None = None,
    ):
        super().__init__()
        self._func = func
        self._params = params
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors or {}
        self._const_buffers: dict[SlotKey, str] = {}
        for i, (key, value) in enumerate(constants.items()):
            buffer_name = f"_const_{i}"
            self.register_buffer(buffer_name, value.detach().clone())
            self._const_buffers[key] = buffer_name

    def forward(self, *dynamic: torch.Tensor) -> torch.Tensor:
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, dynamic))
        for key, buffer_name in self._const_buffers.items():
            slots[key] = getattr(self, buffer_name)
        return self._func(_ReplayEnv(slots, self._sensors), **self._params)


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
    """The ONNX graph input name for a slot.

    Sensor names carry MJCF paths (``robot/imu_lin_vel``), so non-identifier
    characters are folded to ``_``. The authoritative name always travels to the
    runtime in the slot's own ``input`` field (:func:`slot_to_json`) rather than
    being recomputed there, so this scheme stays a build-time detail.
    """
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return "sensor__" + re.sub(r"\W", "_", name_part)
    return f"{namespace}__{name_part}"


def slot_label(key: SlotKey) -> str:
    """Human-readable slot name for diagnostics (parity reports, logs)."""
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return f"sensor:{name_part}"
    return f"{namespace}.{name_part}"


def slot_to_json(key: SlotKey) -> dict[str, Any]:
    """Serialize one input slot for ``policy.json`` / ``config.json``.

    Two shapes, distinguished by which keys are present: ``{"entity", "field"}``
    for an ``Entity.data`` read and ``{"sensor"}`` for a whole-sensor read. Both
    carry ``input`` — the graph input name to feed this slot's value as — so the
    runtime never has to re-derive it from the naming scheme.
    """
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return {"sensor": name_part, "input": _slot_input_name(key)}
    return {"entity": namespace, "field": name_part, "input": _slot_input_name(key)}


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
        namespace, field_name = key
        # A sensor read is live sim state by definition; entity data fields are
        # dynamic only if the registry says so.
        is_dynamic = namespace == _SENSOR_NS or field_name in _DYNAMIC_DATA_FIELDS
        bucket = dynamic if is_dynamic else constants
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
    sensors = dict(recorder._sensors)  # noqa: SLF001 — internal proxy
    module = _TermModule(func, params, dynamic_keys, constants, sensors).eval()
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
    """Read an input slot's current value from ``env`` (entity data or a sensor)."""
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return env.scene[name_part].data
    return getattr(env.scene[namespace].data, name_part)


# ---------------------------------------------------------------------------
# Event terms — side-effecting bodies whose *written* values are traced.
#
# An event term returns None and writes state via ``entity.write_*_to_sim``. We
# capture the tensors it would write (they are functions of the constants and the
# ``rand`` input) and make them the ONNX graph outputs. Randomness is threaded in
# via an explicit ``rand`` input (ADR 0005 §2), replayed by :class:`ReplayRng`.
# ---------------------------------------------------------------------------


# Write kinds an event term may emit, and the ordered field names each produces.
# The "kind" is the mjData write call; "fields" name the tensors it writes, in
# argument order. This is the `entity_write` vocabulary (companion brief §3/§3b).
_WRITE_FIELDS: dict[str, tuple[str, ...]] = {
    "joint_state": ("position", "velocity"),
    "root_pose": ("pose",),
    "root_velocity": ("velocity",),
}


class _WriteCaptureMixin:
    """Records ``write_*_to_sim`` calls into ``self._captures`` (kind → tensors)."""

    def write_joint_state_to_sim(
        self, position, velocity, joint_ids=None, env_ids=None
    ):
        self._captures["joint_state"] = (position, velocity)

    def write_root_link_pose_to_sim(self, pose, env_ids=None):
        self._captures["root_pose"] = (pose,)

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
        self._captures["root_velocity"] = (velocity,)


def _flatten_captures(
    captures: dict[str, tuple[torch.Tensor, ...]],
) -> tuple[list[str], list[torch.Tensor]]:
    """Flatten a captures dict into (output_names, tensors) deterministically.

    Insertion order is the term's own write-call order (stable across runs), so
    the discovery pass and the traced module agree on output ordering.
    """
    names: list[str] = []
    tensors: list[torch.Tensor] = []
    for kind, values in captures.items():
        for field_name, tensor in zip(_WRITE_FIELDS[kind], values):
            names.append(f"{kind}__{field_name}")
            tensors.append(tensor)
    return names, tensors


# A tagged key identifies one value an event body reads off ``env``:
#   ("data", entity, field)  -> entity.data.<field>   (tensor; dynamic or const)
#   ("scene", attr)          -> env.scene.<attr>       (scene-level constant, e.g. env_origins)
#   ("attr", entity, attr)   -> entity.<attr>          (control-flow scalar, e.g. is_fixed_base)
TaggedKey = tuple


class _EvRecData:
    """Records ``entity.data.<field>`` reads as ``("data", entity, field)``."""

    def __init__(self, real, entity, log):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._real, name)
        self._log.append((("data", self._entity, name), value))
        return value


class _EvRecEntity(_WriteCaptureMixin):
    """Records data-field and (non-``data``) attribute reads; captures writes."""

    def __init__(self, real, name, log, captures):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "data", _EvRecData(real.data, name, log))
        object.__setattr__(self, "_captures", captures)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._real, name)
        # Only tensors and control-flow scalars can be reproduced during replay.
        if isinstance(value, (torch.Tensor, bool, int, float)):
            self._log.append((("attr", self._name, name), value))
        return value


class _EvRecScene:
    """Records scene-level attribute reads (e.g. ``env_origins``); indexes entities."""

    def __init__(self, real, log, captures):
        self._real = real
        self._log = log
        self._captures = captures

    def __getitem__(self, name: str) -> _EvRecEntity:
        if _is_sensor(self._real, name):
            # The observation tracer threads sensor reads as slots (`_SENSOR_NS`);
            # the event/command tagged-key path has no equivalent yet, and letting
            # it through would surface as mjlab's own bare `assert isinstance(...)`
            # deep inside the term. Fail here instead, naming the cause.
            raise ValueError(
                f"Event/command term read sensor {name!r}; sensor slots are only "
                "supported for observation/termination terms so far. Extend the "
                "tagged-key proxies (_EvRecScene/_EvReplayScene) the same way "
                "_RecordingScene does, or handle this term natively."
            )
        return _EvRecEntity(self._real[name], name, self._log, self._captures)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._real, name)
        if isinstance(value, (torch.Tensor, bool, int, float)):
            self._log.append((("scene", name), value))
        return value


class _EventCaptureEnv:
    """Proxy env for event tracing: records reads, captures writes, no sim mutation."""

    def __init__(self, real, log, captures):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "scene", _EvRecScene(real.scene, log, captures))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _EvReplayData:
    def __init__(self, entity: str, served: dict[TaggedKey, Any]):
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_served", served)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._served[("data", self._entity, name)]
        except KeyError:
            raise AttributeError(
                f"Event term read undeclared data slot ('data', {self._entity!r}, "
                f"{name!r}) during tracing (input-dependent read?)."
            ) from None


class _EvReplayEntity(_WriteCaptureMixin):
    def __init__(self, entity, served, captures):
        object.__setattr__(self, "_name", entity)
        object.__setattr__(self, "_served", served)
        object.__setattr__(self, "data", _EvReplayData(entity, served))
        object.__setattr__(self, "_captures", captures)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._served[("attr", self._name, name)]
        except KeyError:
            raise AttributeError(
                f"Event term read undeclared attr ('attr', {self._name!r}, {name!r})."
            ) from None


class _EvReplayScene:
    def __init__(self, served, captures):
        self._served = served
        self._captures = captures

    def __getitem__(self, name: str) -> _EvReplayEntity:
        return _EvReplayEntity(name, self._served, self._captures)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._served[("scene", name)]
        except KeyError:
            raise AttributeError(
                f"Event term read undeclared scene attr ('scene', {name!r})."
            ) from None


class _EventReplayEnv:
    def __init__(self, served, captures, num_envs: int = 1, device: str = "cpu"):
        self.scene = _EvReplayScene(served, captures)
        self.num_envs = num_envs
        self.device = device


class _EventModule(nn.Module):
    """Wraps a side-effecting event ``func`` so ``forward(*dynamic, rand)`` returns
    the tensors the term would write, with randomness supplied via ``rand``.

    Dynamic reads arrive as ``forward`` args (data slots that vary at runtime);
    tensor constants are registered buffers; scalar/bool constants (control-flow)
    are held as plain Python values. All are served back through the replay env.
    """

    def __init__(
        self,
        func: Callable[..., None],
        params: dict[str, Any],
        dynamic_keys: list[SlotKey],
        tensor_consts: dict[TaggedKey, torch.Tensor],
        scalar_consts: dict[TaggedKey, Any],
    ):
        super().__init__()
        self._func = func
        self._params = params
        self._dynamic_keys = dynamic_keys
        self._scalar_consts = scalar_consts
        self._const_buffers: dict[TaggedKey, str] = {}
        for i, (key, value) in enumerate(tensor_consts.items()):
            buffer_name = f"_const_{i}"
            self.register_buffer(buffer_name, value.detach().clone())
            self._const_buffers[key] = buffer_name

    def forward(self, *args: torch.Tensor):
        *dynamic, rand = args
        served: dict[TaggedKey, Any] = dict(self._scalar_consts)
        for key, buffer_name in self._const_buffers.items():
            served[key] = getattr(self, buffer_name)
        for (entity, field_name), tensor in zip(self._dynamic_keys, dynamic):
            served[("data", entity, field_name)] = tensor
        captures: dict[str, tuple[torch.Tensor, ...]] = {}
        env = _EventReplayEnv(served, captures)
        with ReplayRng(self._func, rand):
            self._func(env, None, **self._params)
        _, tensors = _flatten_captures(captures)
        return tuple(tensors)


@dataclass
class EventExport:
    """The result of tracing one event term body to ONNX."""

    name: str
    mode: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    input_names: list[str]
    rand_dim: int
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    """Per write-kind descriptor: what the outputs target (entity, kind, fields)."""
    reference_outputs: tuple[torch.Tensor, ...]
    reference_rand: torch.Tensor
    constant_slots: list[str] = field(default_factory=list)


def trace_event_term(
    func: Callable[..., None],
    params: dict[str, Any],
    env: Any,
    *,
    name: str,
    mode: str,
    opset: int = 17,
) -> EventExport:
    """Trace a side-effecting (write-to-sim) event term body to ONNX.

    Supports any combination of ``write_joint_state_to_sim`` (reset joints) and
    ``write_root_link_pose_to_sim`` / ``write_root_link_velocity_to_sim`` (root
    ``entity_write``, companion brief §3/§3b). The written tensors become the
    graph outputs; randomness is supplied via an explicit ``rand`` input recorded
    from the live term. State the term reads off ``env`` is classified into
    dynamic inputs vs baked constants: time-varying ``entity.data`` fields become
    graph inputs; other ``data`` fields, scene-level tensors (``env_origins``),
    and control-flow scalars (``is_fixed_base``) are baked as constants.
    """
    # 1. Discovery on the live env: record draws + reads + written values.
    log: list[tuple[TaggedKey, Any]] = []
    captures: dict[str, tuple[torch.Tensor, ...]] = {}
    proxy = _EventCaptureEnv(env, log, captures)
    with DrawRecorder(func) as rec:
        func(proxy, None, **params)

    if not captures:
        raise ValueError(
            f"Event term {name!r} wrote nothing traceable "
            "(no write_joint_state/root_pose/root_velocity_to_sim call); handle "
            "it natively or extend _WRITE_FIELDS."
        )
    output_names, ref_tensors = _flatten_captures(captures)
    ref_rand = rec.rand_vector
    rand_dim = rec.rand_dim

    # 2. Classify recorded reads: dynamic data-field inputs vs baked constants.
    dynamic: dict[SlotKey, torch.Tensor] = {}
    tensor_consts: dict[TaggedKey, torch.Tensor] = {}
    scalar_consts: dict[TaggedKey, Any] = {}
    for key, value in log:
        is_dynamic = (
            key[0] == "data"
            and key[2] in _DYNAMIC_DATA_FIELDS
            and isinstance(value, torch.Tensor)
        )
        if is_dynamic:
            dynamic.setdefault((key[1], key[2]), value)
        elif isinstance(value, torch.Tensor):
            tensor_consts.setdefault(key, value)
        else:
            scalar_consts.setdefault(key, value)

    dynamic_keys = sorted(dynamic)
    dyn_input_names = [_slot_input_name(k) for k in dynamic_keys]
    example = tuple(dynamic[k] for k in dynamic_keys) + (ref_rand,)
    input_names = [*dyn_input_names, "rand"]

    # 3. Trace: rand replayed as an explicit input; written values captured.
    module = _EventModule(
        func, params, dynamic_keys, tensor_consts, scalar_consts
    ).eval()
    dyn_axes = {n: {0: "batch"} for n in [*dyn_input_names, *output_names]}
    buffer = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            buffer,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dyn_axes,
            opset_version=opset,
            dynamo=False,
        )

    asset_cfg = params.get("asset_cfg")
    entity = getattr(asset_cfg, "name", None)
    write_targets = [
        {
            "kind": kind,
            "entity": entity,
            "fields": list(_WRITE_FIELDS[kind]),
            **(
                {"joint_ids": _static_ids(getattr(asset_cfg, "joint_ids", None))}
                if kind == "joint_state"
                else {}
            ),
        }
        for kind in captures
    ]

    return EventExport(
        name=name,
        mode=mode,
        onnx_bytes=buffer.getvalue(),
        input_slots=dynamic_keys,
        input_names=dyn_input_names,
        rand_dim=rand_dim,
        output_names=output_names,
        write_targets=write_targets,
        reference_outputs=tuple(t.detach() for t in ref_tensors),
        reference_rand=ref_rand.detach(),
        constant_slots=[":".join(str(p) for p in k) for k in sorted(tensor_consts)],
    )


def _static_ids(ids: Any) -> Any:
    if isinstance(ids, slice):
        return "all"
    if hasattr(ids, "tolist"):
        return ids.tolist()
    return ids


# ---------------------------------------------------------------------------
# Command terms — stateful, class-based (ADR 0005 §3, companion brief §3).
#
# A command is a live CommandTerm instance with hidden state (self.<field>). We
# trace `_resample_command` (gated by resample_mask) + `_update_command` (always)
# as a pure function, promoting the hidden state to explicit graph I/O:
#
#     forward(prev_state..., resample_mask, rand) -> (next_state..., entity_write?)
#
# The native orchestrator holds `state` across frames and owns the resample timer
# (ADR 0005 §5). Reset unifies to resample_mask=True.
# ---------------------------------------------------------------------------

_ENTITY_WRITE_METHODS = {
    "write_joint_state_to_sim": "joint_state",
    "write_root_link_pose_to_sim": "root_pose",
    "write_root_link_velocity_to_sim": "root_velocity",
}


def _entity_attrs(term: Any) -> list[str]:
    """Names of ``term`` attributes that are entities (read state from / write to)."""
    return [
        attr
        for attr, value in vars(term).items()
        if hasattr(value, "data")
        and any(hasattr(type(value), m) for m in _ENTITY_WRITE_METHODS)
    ]


class _RecordCommand:
    """Swap a command's entity attrs + ``_env`` to recording proxies (event tagged
    keys) so its reads are logged and its writes captured, with no sim mutation.

    Reused by the tracer's discovery pass and the parity harness's reference run.
    Single-entity commands only: all entity attrs are keyed by ``cfg.entity_name``.
    """

    def __init__(self, term: Any, entity_attr_names: list[str], entity_name: str):
        self.term = term
        self._attrs = entity_attr_names
        self._entity_name = entity_name
        self.log: list[tuple[TaggedKey, Any]] = []
        self.captures: dict[str, tuple[torch.Tensor, ...]] = {}

    def __enter__(self) -> _RecordCommand:
        self._orig = {a: getattr(self.term, a) for a in self._attrs}
        self._orig_env = getattr(self.term, "_env", None)
        for a in self._attrs:
            setattr(
                self.term,
                a,
                _EvRecEntity(self._orig[a], self._entity_name, self.log, self.captures),
            )
        if self._orig_env is not None:
            self.term._env = _EventCaptureEnv(self._orig_env, self.log, self.captures)
        return self

    def __exit__(self, *exc: object) -> None:
        for a, v in self._orig.items():
            setattr(self.term, a, v)
        if self._orig_env is not None:
            self.term._env = self._orig_env


def _snapshot_state(term: Any) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().clone()
        for k, v in vars(term).items()
        if isinstance(v, torch.Tensor)
    }


def _restore_state(term: Any, snap: dict[str, torch.Tensor]) -> None:
    for k, v in snap.items():
        setattr(term, k, v.clone())


def _gate(
    mask: torch.Tensor, resampled: torch.Tensor, prev: torch.Tensor
) -> torch.Tensor:
    shape = [mask.shape[0]] + [1] * (resampled.dim() - 1)
    m = mask.reshape(shape)
    # ONNX Runtime's Where kernel has no bool-branch implementation; select in
    # int64 and cast back so bool state fields (is_*_env) round-trip.
    if resampled.dtype == torch.bool:
        return torch.where(m, resampled.long(), prev.long()).bool()
    return torch.where(m, resampled, prev)


class _CommandModule(nn.Module):
    """Traces a CommandTerm's resample+update as a pure function.

    ``forward(*dynamic_slots, *prev_state, resample_mask, rand)``. Entity attrs and
    ``_env`` are swapped to replay proxies serving dynamic reads (graph inputs) and
    baked constants; state is injected and read back; the resample is gated by
    ``resample_mask`` (``where(mask, resampled, prev)``); ``_update_command`` always
    runs; any ``entity_write`` is captured. Reset unifies to ``resample_mask=True``.
    """

    def __init__(
        self,
        term: Any,
        state_fields: list[str],
        entity_attr_names: list[str],
        entity_name: str,
        *,
        dynamic_keys: list[SlotKey],
        tensor_consts: dict[TaggedKey, torch.Tensor],
        scalar_consts: dict[TaggedKey, Any],
    ):
        super().__init__()
        self._term = term
        self._state_fields = state_fields
        self._entity_attr_names = entity_attr_names
        self._entity_name = entity_name
        self._dynamic_keys = dynamic_keys
        self._scalar_consts = scalar_consts
        self._env_ids = torch.arange(term.num_envs)
        self._const_buffers: dict[TaggedKey, str] = {}
        for i, (key, value) in enumerate(tensor_consts.items()):
            buffer_name = f"_const_{i}"
            self.register_buffer(buffer_name, value.detach().clone())
            self._const_buffers[key] = buffer_name

    def forward(self, *args: torch.Tensor):
        n_dyn = len(self._dynamic_keys)
        n_state = len(self._state_fields)
        dynamic = args[:n_dyn]
        state_inputs = args[n_dyn : n_dyn + n_state]
        resample_mask = args[n_dyn + n_state]
        rand = args[n_dyn + n_state + 1]

        served: dict[TaggedKey, Any] = dict(self._scalar_consts)
        for key, buffer_name in self._const_buffers.items():
            served[key] = getattr(self, buffer_name)
        for (entity, field_name), tensor in zip(self._dynamic_keys, dynamic):
            served[("data", entity, field_name)] = tensor

        captures: dict[str, tuple[torch.Tensor, ...]] = {}
        orig = {a: getattr(self._term, a) for a in self._entity_attr_names}
        orig_env = getattr(self._term, "_env", None)
        for a in self._entity_attr_names:
            setattr(self._term, a, _EvReplayEntity(self._entity_name, served, captures))
        if orig_env is not None:
            self._term._env = _EventReplayEnv(served, captures)
        try:
            prev = {}
            for field_name, value in zip(self._state_fields, state_inputs):
                setattr(self._term, field_name, value)
                prev[field_name] = value.clone()
            with ReplayRng(self._term._resample_command, rand):
                self._term._resample_command(self._env_ids)
                for field_name in self._state_fields:
                    setattr(
                        self._term,
                        field_name,
                        _gate(
                            resample_mask,
                            getattr(self._term, field_name),
                            prev[field_name],
                        ),
                    )
                self._term._update_command()
            outputs = [getattr(self._term, f) for f in self._state_fields]
            _, write_tensors = _flatten_captures(captures)
            return tuple(outputs) + tuple(write_tensors)
        finally:
            for a, v in orig.items():
                setattr(self._term, a, v)
            if orig_env is not None:
                self._term._env = orig_env


@dataclass
class CommandExport:
    """The result of tracing one command term body to ONNX."""

    name: str
    onnx_bytes: bytes
    state_fields: list[dict[str, Any]]
    """Per state field: {name, shape, dtype} — declared in policy.json (§3a)."""
    command_field: str
    input_slots: list[SlotKey]
    input_names: list[str]
    rand_dim: int
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    reference_rand: torch.Tensor


def trace_command_term(
    term: Any,
    state_fields: list[str],
    *,
    name: str,
    command_field: str,
    opset: int = 17,
) -> CommandExport:
    """Trace a stateful CommandTerm to ONNX (companion brief §3).

    Promotes hidden state (``state_fields``) to explicit graph I/O, threads
    randomness through ``rand`` (from ``sample_uniform``; tensor-method RNG like
    ``Tensor.uniform_`` is unsupported — supply a trace-friendly override, brief
    §3a), and threads time-varying runtime reads (``self.robot.data.<field>``) as
    dynamic graph inputs while baking scene-level constants and control-flow
    scalars. Any ``entity_write`` (cube/root pose+velocity) is captured as output.
    """
    entity_attr_names = _entity_attrs(term)
    entity_name = getattr(getattr(term, "cfg", None), "entity_name", None)
    snap = _snapshot_state(term)
    state_example = tuple(getattr(term, f).detach().clone() for f in state_fields)

    # 1. Discovery: swap to recording proxies; log reads, capture writes, spy draws.
    with _RecordCommand(term, entity_attr_names, entity_name) as rec_env:
        with DrawRecorder(term._resample_command) as rec:
            term._resample_command(torch.arange(term.num_envs))
            term._update_command()
        log = list(rec_env.log)
        captures = dict(rec_env.captures)
    ref_rand = rec.rand_vector
    rand_dim = rec.rand_dim
    _restore_state(term, snap)

    output_write_names, _ = _flatten_captures(captures)
    write_targets = [
        {"kind": kind, "entity": entity_name, "fields": list(_WRITE_FIELDS[kind])}
        for kind in captures
    ]

    # 2. Classify reads: dynamic data inputs vs baked tensor/scalar constants.
    dynamic: dict[SlotKey, torch.Tensor] = {}
    tensor_consts: dict[TaggedKey, torch.Tensor] = {}
    scalar_consts: dict[TaggedKey, Any] = {}
    for key, value in log:
        is_dynamic = (
            key[0] == "data"
            and key[2] in _DYNAMIC_DATA_FIELDS
            and isinstance(value, torch.Tensor)
        )
        if is_dynamic:
            dynamic.setdefault((key[1], key[2]), value)
        elif isinstance(value, torch.Tensor):
            tensor_consts.setdefault(key, value)
        else:
            scalar_consts.setdefault(key, value)

    dynamic_keys = sorted(dynamic)
    dyn_names = [_slot_input_name(k) for k in dynamic_keys]
    prev_names = [f"prev_{f}" for f in state_fields]

    # 3. Trace: dynamic + prev_state + resample_mask=True + rand -> next_state + writes.
    mask = torch.ones(term.num_envs, dtype=torch.bool)
    example = (*(dynamic[k] for k in dynamic_keys), *state_example, mask, ref_rand)
    input_names = [*dyn_names, *prev_names, "resample_mask", "rand"]
    output_names = [f"next_{f}" for f in state_fields] + output_write_names

    module = _CommandModule(
        term,
        state_fields,
        entity_attr_names,
        entity_name,
        dynamic_keys=dynamic_keys,
        tensor_consts=tensor_consts,
        scalar_consts=scalar_consts,
    ).eval()
    dyn_axes = {
        n: {0: "batch"}
        for n in [*dyn_names, *prev_names, "resample_mask", *output_names]
    }
    buffer = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            buffer,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dyn_axes,
            opset_version=opset,
            dynamo=False,
        )
    _restore_state(term, snap)

    state_specs = [
        {
            "name": f,
            "shape": list(getattr(term, f).shape),
            "dtype": str(getattr(term, f).dtype).replace("torch.", ""),
        }
        for f in state_fields
    ]

    return CommandExport(
        name=name,
        onnx_bytes=buffer.getvalue(),
        state_fields=state_specs,
        command_field=command_field,
        input_slots=dynamic_keys,
        input_names=dyn_names,
        rand_dim=rand_dim,
        output_names=output_names,
        write_targets=write_targets,
        reference_rand=ref_rand.detach(),
    )
