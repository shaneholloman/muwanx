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
   ONNX graph input) or a model-derived constant (baked into the graph at trace
   time). The split is a small allowlist of the *constants* — everything else is
   dynamic, so an unrecognized field errs toward a graph input rather than being
   silently frozen. This is the only mjlab knowledge this layer carries.
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
from typing import Any, Callable, Sequence

import torch
from torch import nn

from .rng import DrawRecorder, ReplayRng

# ---------------------------------------------------------------------------
# Field classification — the only mjlab knowledge this layer carries.
#
# A field read off ``Entity.data`` is either **baked as a constant** at trace time
# or threaded as a **graph input**. Only the model-derived constants are listed;
# everything else is treated as time-varying.
#
# The allowlist is deliberately this way round. Baking a field that actually varies
# is *silent* corruption — the graph returns trace-time values forever, and nothing
# downstream can tell. Threading a field that is actually constant costs an extra
# graph input the runtime must supply, which fails loudly and immediately. So an
# unrecognized field defaults to dynamic. (This list started as its inverse, which
# silently froze the end-effector position in Lift-Cube-Yam's `ee_to_cube`; the
# parity harness caught it at max|Δ|≈4e-2.)
# ---------------------------------------------------------------------------

_STATIC_DATA_FIELDS: frozenset[str] = frozenset(
    {
        # Model default pose/velocity, and the limits derived from the MJCF.
        "default_joint_pos",
        "default_joint_vel",
        "default_root_state",
        "default_mass",
        "default_inertia",
        "joint_pos_limits",
        "soft_joint_pos_limits",
        "joint_vel_limits",
        "soft_joint_vel_limits",
        "joint_effort_limits",
        "soft_joint_effort_limits",
    }
)


def _is_dynamic_field(field_name: str) -> bool:
    """Whether an ``Entity.data`` field must be threaded as a graph input."""
    return field_name not in _STATIC_DATA_FIELDS


# A slot key identifies one tensor read off the env, as ``(namespace, name)``:
#   (entity_name, data_field)        -> env.scene[entity].data.<field>
#   (_SENSOR_NS, sensor_name)        -> env.scene[sensor].data (a whole BuiltinSensor)
#   (_COMMAND_NS, "cmd.attr")        -> env.command_manager.get_term(cmd).<attr>
SlotKey = tuple[str, str]

_SENSOR_NS = "__sensor__"
"""Namespace marking a :class:`SlotKey` as a sensor read rather than entity data.

mjlab's scene indexes sensors and entities in one ``scene[name]`` namespace, so the
slot key needs its own marker to stay unambiguous. Not a legal mjlab entity name in
practice (mjlab names come from MJCF bodies/sites)."""

_COMMAND_NS = "__command__"
"""Namespace marking a :class:`SlotKey` as a read of another command term's state.

An observation may depend on a command's current value (mjlab's
``object_to_goal_distance`` reads ``command_manager.get_term(name).target_pos``).
The name part is ``"{command_name}.{attr}"``."""


def _class_proxy(real: Any, overrides: dict[str, Any]) -> Any:
    """A stand-in for a live mjlab object that still satisfies ``isinstance`` checks.

    Terms assert on concrete classes (``builtin_sensor`` does
    ``assert isinstance(sensor, BuiltinSensor)``; ``object_to_goal_distance`` does
    the same for ``LiftingCommand``), so the duck-typed proxies used for entities
    are rejected outright. Subclassing the *real* object's own class and sharing
    its ``__dict__`` keeps those checks true — and keeps every unrelated attribute
    working — while replacing only what ``overrides`` names.
    """
    cls = type(real)
    proxy_cls = type(f"_Proxy{cls.__name__}", (cls,), overrides)
    proxy = object.__new__(proxy_cls)
    proxy.__dict__ = real.__dict__
    return proxy


def _sensor_proxy(real: Any, get_data: Callable[[], Any]) -> Any:
    """A sensor stand-in whose ``.data`` comes from ``get_data``."""
    return _class_proxy(real, {"data": property(lambda _self: get_data())})


def _command_proxy(real: Any, on_tensor: Callable[[str, Any], Any]) -> Any:
    """A command-term stand-in routing every tensor attribute through ``on_tensor``.

    Unlike sensors (one ``data`` property) a command's state is a set of plain
    instance attributes (``target_pos``, ``vel_command_b``, …) living in
    ``__dict__``, so ``__getattr__`` never fires for them and ``__getattribute__``
    is the only hook that sees the read. Non-tensor attributes pass through
    untouched, so methods and cfg still behave normally.
    """

    def __getattribute__(self: Any, attr: str) -> Any:  # noqa: N807
        value = object.__getattribute__(self, attr)
        if isinstance(value, torch.Tensor):
            return on_tensor(attr, value)
        return value

    return _class_proxy(real, {"__getattribute__": __getattribute__})


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
        if isinstance(value, torch.Tensor):
            # A builtin sensor is one `sensordata` window — one slot.
            self._log.append(((_SENSOR_NS, name), value))
            return value
        # A structured sensor (mjlab's `RayCastSensor`: distances, hit_pos_w, …)
        # has no single tensor to be. Log the *fields* the term actually touches,
        # so each becomes its own slot instead of the term looking untraceable.
        return _RecordingSensorData(value, name, self._log)


class _RecordingSensorData:
    """Wraps a structured sensor's ``.data``, logging each tensor field read."""

    def __init__(self, real: Any, sensor: str, log: list[tuple[SlotKey, Any]]):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_sensor", sensor)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._real, name)
        if isinstance(value, torch.Tensor):
            self._log.append(((_SENSOR_NS, f"{self._sensor}.{name}"), value))
        return value


class _RecordingCommandManager:
    """Wraps the real ``CommandManager``, logging command-state tensor reads."""

    def __init__(
        self,
        real: Any,
        log: list[tuple[SlotKey, Any]],
        commands: dict[str, Any],
    ):
        self._real = real
        self._log = log
        self._commands = commands

    def get_term(self, name: str) -> Any:
        real = self._real.get_term(name)
        # Keep the real term so the replay pass can subclass its class.
        self._commands[name] = real

        def on_tensor(attr: str, value: Any) -> Any:
            self._log.append(((_COMMAND_NS, f"{name}.{attr}"), value))
            return value

        return _command_proxy(real, on_tensor)

    def get_command(self, name: str) -> Any:
        value = self._real.get_command(name)
        self._log.append(((_COMMAND_NS, f"{name}.command"), value))
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _RecordingEnv:
    """Proxy env recording the reads a term makes (entity data, sensors, commands)."""

    def __init__(self, real: Any):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_log", [])
        object.__setattr__(self, "_sensors", {})
        object.__setattr__(self, "_commands", {})
        object.__setattr__(
            self, "scene", _RecordingScene(real.scene, self._log, self._sensors)
        )

    def __getattr__(self, name: str) -> Any:
        if name == "command_manager":
            return _RecordingCommandManager(
                self._real.command_manager, self._log, self._commands
            )
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


class _ReplaySensorData:
    """Serves a structured sensor's recorded fields during the replay pass."""

    def __init__(self, sensor: str, slots: dict[SlotKey, torch.Tensor]):
        object.__setattr__(self, "_sensor", sensor)
        object.__setattr__(self, "_slots", slots)

    def __getattr__(self, name: str) -> torch.Tensor:
        key = (_SENSOR_NS, f"{self._sensor}.{name}")
        if key not in self._slots:
            raise AttributeError(
                f"sensor field {self._sensor}.{name} was not recorded during discovery"
            )
        return self._slots[key]


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
            whole = (_SENSOR_NS, name)
            if whole in self._slots:
                return _sensor_proxy(real, lambda: self._slots[whole])
            # Structured sensor: the discovery pass recorded its fields separately.
            return _sensor_proxy(real, lambda: _ReplaySensorData(name, self._slots))
        return _ReplayEntity(name, self._slots)


class _ReplayCommandManager:
    """Serves recorded command-state slots back during tracing."""

    def __init__(self, slots: dict[SlotKey, torch.Tensor], commands: dict[str, Any]):
        self._slots = slots
        self._commands = commands

    def get_term(self, name: str) -> Any:
        real = self._commands.get(name)
        if real is None:
            raise AttributeError(
                f"Term read command {name!r} during tracing that the discovery pass "
                "never saw — the term's control flow is input-dependent, which is "
                "not traceable (ADR 0005 §Consequences)."
            )
        return _command_proxy(
            real, lambda attr, _v: self._slots[(_COMMAND_NS, f"{name}.{attr}")]
        )

    def get_command(self, name: str) -> torch.Tensor:
        return self._slots[(_COMMAND_NS, f"{name}.command")]


class _ReplayEnv:
    def __init__(
        self,
        slots: dict[SlotKey, torch.Tensor],
        sensors: dict[str, Any] | None = None,
        commands: dict[str, Any] | None = None,
    ):
        self.scene = _ReplayScene(slots, sensors)
        self.command_manager = _ReplayCommandManager(slots, commands or {})


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
        *,
        sensors: dict[str, Any] | None = None,
        commands: dict[str, Any] | None = None,
    ):
        super().__init__()
        self._func = func
        self._params = params
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors or {}
        self._commands = commands or {}
        self._const_buffers: dict[SlotKey, str] = {}
        for i, (key, value) in enumerate(constants.items()):
            buffer_name = f"_const_{i}"
            self.register_buffer(buffer_name, value.detach().clone())
            self._const_buffers[key] = buffer_name

    def forward(self, *dynamic: torch.Tensor) -> torch.Tensor:
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, dynamic))
        for key, buffer_name in self._const_buffers.items():
            slots[key] = getattr(self, buffer_name)
        env = _ReplayEnv(slots, self._sensors, self._commands)
        return self._func(env, **self._params)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConstantTerm(ValueError):
    """A term that read no simulation state at all — its value is a constant.

    Genuinely env-independent (a fixed-size padding term, say), so a caller may
    safely bake the value. Distinct from :class:`UntraceableTerm` because the two
    look identical from "no graph inputs" alone and must not be handled alike.
    """


class UntraceableTerm(ValueError):
    """A term that read state the tracer could not follow into the graph.

    The recording pass saw accesses but none yielded a tensor — e.g. mjlab's
    ``height_scan`` reads a ``RayCastSensor`` whose ``.data`` is a dataclass of
    ray hits, not a tensor field. Such a term is *time-varying*, so baking its
    trace-time value freezes it: a policy would receive a fixed terrain profile
    forever, with nothing in the build output saying so. ADR 0005's rule applies —
    a term that fails to trace fails the build.
    """

    def __init__(self, term: str, touched: list[str]):
        self.term = term
        self.touched = touched
        super().__init__(
            f"Observation term {term!r} reads state the tracer cannot turn into a "
            f"graph input: {', '.join(touched) or '(nothing usable)'}. Baking its "
            "current value would freeze a time-varying input and silently feed the "
            "policy stale numbers. Either supply a trace-friendly replacement via "
            "register_observation(), or drop the term from the exported group and "
            "retrain — a shorter observation vector is not interchangeable with the "
            "one the policy was trained on."
        )


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
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots`` (see :func:`slots_json`)."""

    @property
    def is_dynamic_only(self) -> bool:
        return len(self.input_slots) > 0


def _slot_input_name(key: SlotKey) -> str:
    """The ONNX graph input name for a slot.

    Sensor names carry MJCF paths (``robot/imu_lin_vel``) and command slots embed a
    dotted ``cmd.attr``, so non-identifier characters are folded to ``_``. The
    authoritative name always travels to the runtime in the slot's own ``input``
    field (:func:`slot_to_json`) rather than being recomputed there, so this scheme
    stays a build-time detail.
    """
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return "sensor__" + re.sub(r"\W", "_", name_part)
    if namespace == _COMMAND_NS:
        return "command__" + re.sub(r"\W", "_", name_part)
    return f"{namespace}__{name_part}"


def slot_label(key: SlotKey) -> str:
    """Human-readable slot name for diagnostics (parity reports, logs)."""
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return f"sensor:{name_part}"
    if namespace == _COMMAND_NS:
        return f"command:{name_part}"
    return f"{namespace}.{name_part}"


def slot_to_json(key: SlotKey, shape: Sequence[int] | None = None) -> dict[str, Any]:
    """Serialize one input slot for ``policy.json`` / ``config.json``.

    Three shapes, distinguished by which keys are present: ``{"entity", "field"}``
    for an ``Entity.data`` read, ``{"sensor"}`` for a whole-sensor read, and
    ``{"command", "field"}`` for another command term's state. All carry
    ``input`` — the graph input name to feed this slot's value as — so the
    runtime never has to re-derive it from the naming scheme.

    ``shape`` is the traced tensor's shape, batch axis included. The runtime feeds
    a flat value array, so without it there is nothing to reconstruct the rank
    from and it can only guess ``(batch, n)`` — which ORT rejects outright for the
    fields that aren't rank 2: ``site_pos_w`` is ``(batch, num_sites, 3)`` and
    ``heading_w`` is ``(batch,)``.
    """
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        sensor_name, dot, sensor_field = name_part.partition(".")
        entry = {"sensor": sensor_name, "input": _slot_input_name(key)}
        if dot:
            # A structured sensor (mjlab's RayCastSensor) contributes one slot per
            # field the term reads, rather than one window of `sensordata`.
            entry["field"] = sensor_field
    elif namespace == _COMMAND_NS:
        command_name, _, attr = name_part.partition(".")
        entry = {
            "command": command_name,
            "field": attr,
            "input": _slot_input_name(key),
        }
    else:
        entry = {
            "entity": namespace,
            "field": name_part,
            "input": _slot_input_name(key),
        }
    if shape is not None:
        entry["shape"] = [int(d) for d in shape]
    return entry


def slots_json(export: Any) -> list[dict[str, Any]]:
    """Serialize the input slots the exported graph actually takes, shapes included.

    Shared by all three export kinds so the wire format can only be described in
    one place. ``input_shapes`` is positionally parallel to ``input_slots``; a
    short or absent list degrades to shape-less entries rather than raising, which
    keeps hand-built exports in tests usable.

    Slots the exporter folded away are dropped rather than emitted: a term reading
    an integer index tensor (mjlab's ``MotionCommand.body_indexes``) declares it as
    a slot, and ``torch.onnx.export`` then bakes it into the Gather it feeds. The
    runtime feeds by name, and ORT rejects a feed that is not a graph input
    outright (``invalid input '…'``), so keeping the slot would break every run of
    the graph — and there would be nothing to read it from either.
    """
    shapes = getattr(export, "input_shapes", None) or []
    entries = [
        slot_to_json(key, shapes[i] if i < len(shapes) else None)
        for i, key in enumerate(export.input_slots)
    ]
    graph_inputs = _graph_input_names(getattr(export, "onnx_bytes", None))
    if graph_inputs is None:
        return entries
    return [entry for entry in entries if entry["input"] in graph_inputs]


def _graph_input_names(onnx_bytes: bytes | None) -> set[str] | None:
    """Input names of an exported graph, or None when there is no graph to ask.

    Unparseable bytes answer None rather than raising: a hand-built export in a
    test carries a placeholder, and every real one came out of
    ``torch.onnx.export`` a few lines earlier. Filtering nothing degrades to the
    pre-filter behaviour; refusing to serialize the term does not.
    """
    if not onnx_bytes:
        return None
    import onnx

    try:
        model = onnx.load_from_string(onnx_bytes)
    except Exception:
        return None
    return {i.name for i in model.graph.input}


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
        # Sensor and command-state reads are live state by definition; entity data
        # fields are dynamic unless they are model-derived constants.
        is_dynamic = namespace in (_SENSOR_NS, _COMMAND_NS) or _is_dynamic_field(
            field_name
        )
        bucket = dynamic if is_dynamic else constants
        bucket.setdefault(key, value)

    if not dynamic:
        if recorder._log:  # noqa: SLF001 — internal proxy
            # State *was* read; the tracer just could not follow it into a tensor.
            raise UntraceableTerm(
                name, sorted({slot_label(k) for k, _ in recorder._log})
            )  # noqa: SLF001
        raise ConstantTerm(
            f"Term {name!r} reads no simulation state at all; handle it as a native "
            "term (e.g. time_out) or bake its value (ADR 0005)."
        )

    dynamic_keys = sorted(dynamic)
    input_names = [_slot_input_name(k) for k in dynamic_keys]
    example_inputs = tuple(dynamic[k] for k in dynamic_keys)

    # 3. Trace to ONNX.
    sensors = dict(recorder._sensors)  # noqa: SLF001 — internal proxy
    commands = dict(recorder._commands)  # noqa: SLF001 — internal proxy
    module = _TermModule(
        func, params, dynamic_keys, constants, sensors=sensors, commands=commands
    ).eval()
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
        input_shapes=[list(t.shape) for t in example_inputs],
    )


def read_slot(env: Any, key: SlotKey) -> torch.Tensor:
    """Read an input slot's current value from ``env``.

    Handles all three slot namespaces: entity data, a whole sensor, and another
    command term's state.
    """
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        sensor_name, dot, sensor_field = name_part.partition(".")
        data = env.scene[sensor_name].data
        return getattr(data, sensor_field) if dot else data
    if namespace == _COMMAND_NS:
        command_name, _, attr = name_part.partition(".")
        return getattr(env.command_manager.get_term(command_name), attr)
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
    rand_ranges: list[list[float]]
    """Per-element ``[low, high]`` for ``rand`` — the runtime draws with these."""
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    """Per write-kind descriptor: what the outputs target (entity, kind, fields)."""
    reference_outputs: tuple[torch.Tensor, ...]
    reference_rand: torch.Tensor
    constant_slots: list[str] = field(default_factory=list)
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots`` (see :func:`slots_json`)."""


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
    rand_ranges = rec.rand_ranges

    # 2. Classify recorded reads: dynamic data-field inputs vs baked constants.
    dynamic: dict[SlotKey, torch.Tensor] = {}
    tensor_consts: dict[TaggedKey, torch.Tensor] = {}
    scalar_consts: dict[TaggedKey, Any] = {}
    for key, value in log:
        is_dynamic = (
            key[0] == "data"
            and _is_dynamic_field(key[2])
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
        rand_ranges=rand_ranges,
        output_names=output_names,
        write_targets=write_targets,
        reference_outputs=tuple(t.detach() for t in ref_tensors),
        reference_rand=ref_rand.detach(),
        constant_slots=[":".join(str(p) for p in k) for k in sorted(tensor_consts)],
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
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
    rand_ranges: list[list[float]]
    """Per-element ``[low, high]`` for ``rand`` — the runtime draws with these."""
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    reference_rand: torch.Tensor
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots`` (see :func:`slots_json`)."""


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
    rand_ranges = rec.rand_ranges
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
            and _is_dynamic_field(key[2])
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

    # `init` is ADR 0005 §3's "names, shapes, *and initial values*". The state was
    # restored from `snap` above, so these are the values `cfg.build(env)` left —
    # the term's state before any resample. Without them the runtime zero-fills,
    # which is right only for a term whose first resample overwrites every field;
    # a counter or a held previous value would start wrong, silently.
    state_specs = [
        {
            "name": f,
            "shape": list(getattr(term, f).shape),
            "dtype": str(getattr(term, f).dtype).replace("torch.", ""),
            "init": [
                # bool/int state round-trips as a number; the reader rebuilds dtype.
                bool(v) if getattr(term, f).dtype == torch.bool else v
                for v in getattr(term, f).detach().reshape(-1).tolist()
            ],
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
        rand_ranges=rand_ranges,
        output_names=output_names,
        write_targets=write_targets,
        reference_rand=ref_rand.detach(),
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
    )


# ---------------------------------------------------------------------------
# Observation-group fusion (ADR 0005 §4, companion brief §4b)
# ---------------------------------------------------------------------------


NATIVE_OBSERVATION_FUNCS: dict[str, str] = {
    # mjlab funcs that read env-level state rather than `entity.data`, so there is
    # nothing to trace: the runtime already holds these values every frame.
    "last_action": "prev_action",
    "generated_commands": "command",
}


def _native_observation_kind(func: Callable[..., Any]) -> str | None:
    return NATIVE_OBSERVATION_FUNCS.get(getattr(func, "__name__", ""))


@dataclass
class GroupTermSpec:
    """One term inside a fused group, as :func:`trace_observation_group` needs it."""

    name: str
    func: Callable[..., torch.Tensor]
    params: dict[str, Any]
    clip: tuple[float, float] | None = None
    scale: Any = None
    """Per-term scale — a float, or a sequence broadcast over the term's width."""


@dataclass
class GroupExport:
    """The result of fusing one observation group into a single ONNX graph."""

    name: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    """Deduplicated union of every term's dynamic slots, in graph input order."""
    input_names: list[str]
    input_shapes: list[list[int]]
    native_inputs: list[dict[str, Any]]
    """Per native term: ``{name, native, input, size, ...}`` — fed by the runtime."""
    layout: list[dict[str, Any]]
    """``{name, size}`` per term, in concat order, for the runtime's group layout."""
    output_name: str
    reference_output: torch.Tensor
    constant_slots: list[SlotKey] = field(default_factory=list)


class _GroupModule(nn.Module):
    """Runs a whole observation group: every term body, then clip/scale, then cat.

    The single ``forward`` reproduces mjlab's ``compute_group`` for the terms it
    owns — per-term ``clip`` *then* ``scale`` (that order is mjlab's), then
    concatenation in declaration order. One replay env is built for all of them, so
    a slot two terms share is read once rather than marshalled twice.

    Native terms are graph *inputs* rather than bodies: ``last_action`` and
    ``generated_commands`` read env-level state the runtime already holds, so
    feeding the value in keeps the group's output the complete observation vector
    instead of something the runtime must splice offsets into.
    """

    def __init__(
        self,
        terms: list[GroupTermSpec],
        dynamic_keys: list[SlotKey],
        constants: dict[SlotKey, torch.Tensor],
        *,
        sensors: dict[str, Any],
        commands: dict[str, Any],
        native_names: list[str],
        baked: dict[str, torch.Tensor],
    ):
        super().__init__()
        self._terms = terms
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors
        self._commands = commands
        self._native_names = native_names
        self._const_buffers: dict[SlotKey, str] = {}
        for i, (key, value) in enumerate(constants.items()):
            buffer_name = f"_const_{i}"
            self.register_buffer(buffer_name, value.detach().clone())
            self._const_buffers[key] = buffer_name
        # Terms with no dynamic state at all (a fixed-size padding term, say) are
        # values, not functions — bake them like any other constant.
        self._baked_buffers: dict[str, str] = {}
        for i, (term_name, value) in enumerate(baked.items()):
            buffer_name = f"_baked_{i}"
            self.register_buffer(buffer_name, value.detach().clone())
            self._baked_buffers[term_name] = buffer_name

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        split = len(self._dynamic_keys)
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, args[:split]))
        for key, buffer_name in self._const_buffers.items():
            slots[key] = getattr(self, buffer_name)
        native = dict(zip(self._native_names, args[split:]))
        env = _ReplayEnv(slots, self._sensors, self._commands)

        pieces: list[torch.Tensor] = []
        for term in self._terms:
            if term.name in native:
                value = native[term.name]
            elif term.name in self._baked_buffers:
                value = getattr(self, self._baked_buffers[term.name])
            else:
                value = term.func(env, **term.params)
            # mjlab's order: clip, then scale (observation_manager.compute_group).
            if term.clip is not None:
                value = torch.clamp(value, min=term.clip[0], max=term.clip[1])
            if term.scale is not None:
                value = value * _scale_tensor(term.scale, value)
            pieces.append(value.reshape(value.shape[0], -1))
        return torch.cat(pieces, dim=-1)


def _scale_tensor(scale: Any, like: torch.Tensor) -> torch.Tensor:
    """A term's ``scale`` as a tensor broadcastable over its output."""
    if isinstance(scale, torch.Tensor):
        return scale.to(like.dtype)
    if isinstance(scale, (list, tuple)):
        return torch.tensor(list(scale), dtype=like.dtype, device=like.device)
    return torch.tensor(float(scale), dtype=like.dtype, device=like.device)


def trace_observation_group(
    terms: list[GroupTermSpec],
    env: Any,
    *,
    name: str,
    opset: int = 17,
) -> GroupExport:
    """Fuse an observation group's terms into one ONNX graph (ADR 0005 §4).

    One graph per group instead of one per term. The motivation is measured in the
    companion brief §4b: a per-term graph can be a *single* node (three of G1's
    five are `Identity`), so the fixed per-``ort.run()`` cost — the JS↔WASM
    crossing, tensor marshalling, a promise round-trip — is the entire expense, and
    slots two terms share get marshalled twice.

    Inputs are the deduplicated union of the terms' dynamic slots, followed by one
    input per native term. The output is the group's concatenated vector with each
    term's clip/scale folded in — i.e. exactly what the policy consumes, minus
    history (state across frames, which stays with the runtime's ring buffer).

    Raises:
        ValueError: if no term reads dynamic state (the whole group is constant, so
            there is nothing to run per frame) or if tracing fails.
    """
    # 1. Discovery, per term: what does each read, and is it native?
    dynamic: dict[SlotKey, torch.Tensor] = {}
    constants: dict[SlotKey, torch.Tensor] = {}
    sensors: dict[str, Any] = {}
    commands: dict[str, Any] = {}
    native_inputs: list[dict[str, Any]] = []
    native_examples: list[torch.Tensor] = []
    baked: dict[str, torch.Tensor] = {}
    layout: list[dict[str, Any]] = []

    for term in terms:
        native_kind = _native_observation_kind(term.func)
        if native_kind is not None:
            value = term.func(env, **term.params).detach()
            entry: dict[str, Any] = {
                "name": term.name,
                "native": native_kind,
                "input": "native__" + re.sub(r"\W", "_", term.name),
                "size": int(value.reshape(1, -1).shape[-1]),
            }
            if native_kind == "command":
                entry["command_name"] = term.params["command_name"]
            elif term.params.get("action_name") is not None:
                entry["action_name"] = term.params["action_name"]
            native_inputs.append(entry)
            native_examples.append(value)
            layout.append({"name": term.name, "size": entry["size"]})
            continue

        recorder = _RecordingEnv(env)
        recorded = term.func(recorder, **term.params)
        if not isinstance(recorded, torch.Tensor):
            raise ValueError(
                f"Observation term {term.name!r} returned "
                f"{type(recorded).__name__}, not a Tensor."
            )
        term_dynamic = False
        for key, value in recorder._log:  # noqa: SLF001 — internal proxy
            if not isinstance(value, torch.Tensor):
                continue
            namespace, field_name = key
            if namespace in (_SENSOR_NS, _COMMAND_NS) or _is_dynamic_field(field_name):
                dynamic.setdefault(key, value)
                term_dynamic = True
            else:
                constants.setdefault(key, value)
        sensors.update(recorder._sensors)  # noqa: SLF001 — internal proxy
        commands.update(recorder._commands)  # noqa: SLF001 — internal proxy
        if not term_dynamic:
            # Same discriminator as `trace_term`: nothing read at all means a
            # genuine constant; reads the tracer could not follow mean a
            # time-varying term that must not be frozen into the group's vector.
            if recorder._log:  # noqa: SLF001 — internal proxy
                raise UntraceableTerm(
                    term.name,
                    sorted({slot_label(k) for k, _ in recorder._log}),  # noqa: SLF001
                )
            baked[term.name] = recorded.detach()
        layout.append(
            {"name": term.name, "size": int(recorded.reshape(1, -1).shape[-1])}
        )

    if not dynamic:
        raise ValueError(
            f"Observation group {name!r} reads no time-varying state; every term is "
            "native or constant, so there is no graph to run."
        )

    # 2. Fuse and export. Slot order is sorted for determinism; native inputs
    #    follow, in declaration order.
    dynamic_keys = sorted(dynamic)
    slot_names = [_slot_input_name(k) for k in dynamic_keys]
    native_names = [entry["name"] for entry in native_inputs]
    input_names = [*slot_names, *(entry["input"] for entry in native_inputs)]
    example_inputs = tuple(dynamic[k] for k in dynamic_keys) + tuple(native_examples)

    module = _GroupModule(
        terms,
        dynamic_keys,
        constants,
        sensors=sensors,
        commands=commands,
        native_names=native_names,
        baked=baked,
    ).eval()
    output_name = "obs"
    buffer = io.BytesIO()
    with torch.no_grad():
        reference = module(*example_inputs).detach()
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

    return GroupExport(
        name=name,
        onnx_bytes=buffer.getvalue(),
        input_slots=dynamic_keys,
        input_names=slot_names,
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
        native_inputs=native_inputs,
        layout=layout,
        output_name=output_name,
        reference_output=reference,
        constant_slots=sorted(constants),
    )


# ---------------------------------------------------------------------------
# Termination-group fusion (ADR 0005 §4, companion brief §4b)
# ---------------------------------------------------------------------------


@dataclass
class TerminationGroupExport:
    """The result of fusing a set of termination terms into a single ONNX graph."""

    name: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    input_names: list[str]
    input_shapes: list[list[int]]
    lanes: list[str]
    """Term names, in output-lane order — lane *i* is `lanes[i]`'s verdict."""
    output_name: str
    reference_output: torch.Tensor
    constant_slots: list[SlotKey] = field(default_factory=list)


class _TerminationGroupModule(nn.Module):
    """Every termination body in one graph, emitting one bool lane per term.

    A lane rather than a single OR because the manager reports *which* term fired
    (its `reasons`) and splits `time_out` from real terminations — collapsing them
    here would throw that away to save one comparison.
    """

    def __init__(
        self,
        terms: list[GroupTermSpec],
        dynamic_keys: list[SlotKey],
        constants: dict[SlotKey, torch.Tensor],
        *,
        sensors: dict[str, Any],
        commands: dict[str, Any],
    ):
        super().__init__()
        self._terms = terms
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors
        self._commands = commands
        self._const_buffers: dict[SlotKey, str] = {}
        for i, (key, value) in enumerate(constants.items()):
            buffer_name = f"_const_{i}"
            self.register_buffer(buffer_name, value.detach().clone())
            self._const_buffers[key] = buffer_name

    def forward(self, *dynamic: torch.Tensor) -> torch.Tensor:
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, dynamic))
        for key, buffer_name in self._const_buffers.items():
            slots[key] = getattr(self, buffer_name)
        env = _ReplayEnv(slots, self._sensors, self._commands)
        lanes = [term.func(env, **term.params).reshape(-1, 1) for term in self._terms]
        return torch.cat(lanes, dim=-1)


def trace_termination_group(
    terms: list[GroupTermSpec],
    env: Any,
    *,
    name: str,
    opset: int = 17,
) -> TerminationGroupExport:
    """Fuse termination terms into one graph, one bool lane each (ADR 0005 §4).

    Same motivation as observation fusion (companion brief §4b) and the same
    mechanics — the deduplicated union of the terms' slots in, one graph out —
    but the output is a bool *vector*, one lane per term, so the manager keeps
    per-term `reasons` and its terminated-vs-truncated split.

    `time_out` never reaches here: it reads no entity state, so it is classified
    native before this is called.

    Raises:
        ValueError: if no term reads dynamic state, or if tracing fails.
    """
    dynamic: dict[SlotKey, torch.Tensor] = {}
    constants: dict[SlotKey, torch.Tensor] = {}
    sensors: dict[str, Any] = {}
    commands: dict[str, Any] = {}

    for term in terms:
        recorder = _RecordingEnv(env)
        recorded = term.func(recorder, **term.params)
        if not isinstance(recorded, torch.Tensor):
            raise ValueError(
                f"Termination term {term.name!r} returned "
                f"{type(recorded).__name__}, not a Tensor."
            )
        term_dynamic = False
        for key, value in recorder._log:  # noqa: SLF001 — internal proxy
            if not isinstance(value, torch.Tensor):
                continue
            namespace, field_name = key
            if namespace in (_SENSOR_NS, _COMMAND_NS) or _is_dynamic_field(field_name):
                dynamic.setdefault(key, value)
                term_dynamic = True
            else:
                constants.setdefault(key, value)
        sensors.update(recorder._sensors)  # noqa: SLF001 — internal proxy
        commands.update(recorder._commands)  # noqa: SLF001 — internal proxy
        if not term_dynamic:
            # Unlike an observation there is no constant to bake: a termination
            # that cannot see time-varying state either never fires or always
            # does, and either way the build should say so rather than emit it.
            raise UntraceableTerm(
                term.name,
                sorted({slot_label(k) for k, _ in recorder._log}),  # noqa: SLF001
            )

    if not dynamic:
        raise ValueError(
            f"Termination group {name!r} reads no time-varying state; every term "
            "should be native (e.g. time_out)."
        )

    dynamic_keys = sorted(dynamic)
    input_names = [_slot_input_name(k) for k in dynamic_keys]
    example_inputs = tuple(dynamic[k] for k in dynamic_keys)

    module = _TerminationGroupModule(
        terms, dynamic_keys, constants, sensors=sensors, commands=commands
    ).eval()
    output_name = "done"
    buffer = io.BytesIO()
    with torch.no_grad():
        reference = module(*example_inputs).detach()
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

    return TerminationGroupExport(
        name=name,
        onnx_bytes=buffer.getvalue(),
        input_slots=dynamic_keys,
        input_names=input_names,
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
        lanes=[term.name for term in terms],
        output_name=output_name,
        reference_output=reference,
        constant_slots=sorted(constants),
    )
