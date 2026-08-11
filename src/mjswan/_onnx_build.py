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

import copy
import inspect
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .command import ButtonConfig, CommandTermConfig
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


def _require_ts_src(kind: str, name: str, binding: Any) -> None:
    """A ``*Binding`` without ``ts_src`` names a class the browser does not have.

    mjswan ships no built-in TS observation/termination/event classes — every
    built-in term is a traced graph or a native marker — so a binding is only ever
    the custom-TS escape hatch. Without the file there is nothing to run, and the
    term would go missing from a bundle that reports itself as complete.
    """
    if binding.ts_src:
        return
    raise ValueError(
        f"{kind} term {name!r} is bound to TS class {binding.ts_name or '(unnamed)'!r} "
        "but no `ts_src` was given, so the browser has no implementation to run: "
        "mjswan ships no built-in TS term classes. Either let the build trace the "
        f"term's own function, or point `ts_src` at a `.ts` file exporting "
        f"{binding.ts_name or 'the class'!r}."
    )


def _write_onnx(out_dir: Path, ref: str, onnx_bytes: bytes) -> None:
    path = out_dir / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_bytes)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def _tensor_width(value: Any) -> int:
    """Per-env element count of a term's output (batch axis folded away)."""
    return int(value.detach().reshape(1, -1).shape[-1])


def _resolved_params(params: dict[str, Any], env: Any) -> dict[str, Any]:
    """Resolve every ``SceneEntityCfg`` in *params* against the live scene.

    mjlab's managers do this once at ``_prepare_terms``, turning name patterns
    into concrete indices (``site_names=('grasp_site',)`` → ``site_ids=[1]``);
    the term bodies then index with those ids. The Builder serializes from the
    *task config*, whose ``SceneEntityCfg``s are still unresolved
    (``site_ids=slice(None)`` — i.e. *every* site), so tracing without this step
    bakes a different function than mjlab actually runs. Lift-Cube-Yam's
    ``ee_to_cube`` returned all 2 sites (6 values) instead of the grasp site
    (3), which would have fed 6 wrong numbers to a policy trained on 3.

    Resolution mutates the cfg, so a copy is resolved and the caller's config is
    left untouched. Duck-typed rather than ``isinstance``-checked to keep mjlab a
    soft dependency.
    """
    resolved = dict(params)
    for key, value in params.items():
        if callable(getattr(value, "resolve", None)) and hasattr(value, "name"):
            entity_cfg = copy.deepcopy(value)
            entity_cfg.resolve(env.scene)
            resolved[key] = entity_cfg
    return resolved


def _native_observation_entry(
    name: str, func: Any, params: dict[str, Any], env: Any
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

    ``size`` is attached when the live env can supply it, since the runtime sizes
    its observation buffers before the first step. It is best-effort here: a scene
    may pair ``generated_commands`` with a command that only exists browser-side
    (a native ``UiCommand``), in which case mjlab's own lookup raises and the
    runtime resolves the width from the command itself instead.

    ``action_offset`` rides along when ``last_action`` names a term, because then it
    is that term's slice of the policy output rather than the whole vector, and the
    runtime holds the vector whole (see :func:`~mjswan.compile.tracer.
    action_term_offset`). Unlike ``size`` it is *not* best-effort: falling back to the
    whole vector is the silently-wrong observation it exists to prevent.
    """
    from .compile.tracer import action_term_offset

    func_name = getattr(func, "__name__", None)
    if func_name == "last_action":
        entry: dict[str, Any] = {"name": name, "native": "prev_action"}
        action_name = params.get("action_name")
        if action_name is not None:
            entry["action_name"] = action_name
            # Outside the `size` probe below, whose swallowed failure would lose this slice.
            entry["action_offset"] = action_term_offset(env, action_name)
    elif func_name == "generated_commands":
        entry = {
            "name": name,
            "native": "command",
            "command_name": params["command_name"],
        }
    else:
        return None

    try:
        width = _tensor_width(func(env, **params))
    except Exception:  # noqa: BLE001 — best-effort; runtime resolves it instead
        width = 0
    if width:
        # A zero width means "no action manager", not a term of no width.
        entry["size"] = width
    return entry


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
    if term_cfg.history_steps:
        # Sparse offsets are per-term, so a group count would be a second answer.
        entry["history_offsets"] = [int(step) for step in term_cfg.history_steps]
    elif history:
        entry["history_length"] = history
    # Interleaving describes a stack's layout, so it says nothing without a stack.
    if term_cfg.history_interleaved and (
        "history_offsets" in entry or "history_length" in entry
    ):
        entry["history_interleaved"] = True
    return entry


def serialize_observation_term(
    name: str,
    term_cfg: ObservationTermCfg,
    env: Any,
    out_dir: Path,
    group_history_length: int | None,
) -> dict[str, Any] | None:
    """Serialize one observation term.

    Raises rather than degrading. An observation is part of the policy's input
    vector, so every way of *not* emitting a term correctly produces a silently
    wrong policy: dropping it shortens the vector the network was trained on, and
    baking a time-varying term freezes an input. Both used to happen here — see
    :class:`~mjswan.compile.tracer.UntraceableTerm`.
    """
    from .compile import trace_term
    from .compile.tracer import ConstantTerm, slots_json

    func = term_cfg.func
    if isinstance(func, ObservationBinding):
        _require_ts_src("Observation", name, func)
        return term_cfg.to_dict()

    params = _resolved_params(term_cfg.params, env)

    native_entry = _native_observation_entry(name, func, params, env)
    if native_entry is not None:
        return _apply_observation_pipeline(native_entry, term_cfg, group_history_length)

    try:
        export = trace_term(func, params, env, name=name)
    except ConstantTerm:
        # Reads nothing off the env, so bake it from a real call. Not `UntraceableTerm`,
        # where state *was* read and merely not followed.
        import torch

        value = func(env, **params)
        if not isinstance(value, torch.Tensor):
            raise
        values = value.detach().flatten().tolist()
        entry = {
            "name": name,
            "native": "constant",
            "value": values,
            "size": len(values),
        }
        return _apply_observation_pipeline(entry, term_cfg, group_history_length)
    ref = _onnx_ref("obs", name)
    _write_onnx(out_dir, ref, export.onnx_bytes)

    # The runtime cannot infer `size`: inference is async, the group layout is not.
    entry = {
        "name": name,
        "onnx": ref,
        "size": _tensor_width(export.reference_output),
        "input_slots": slots_json(export),
    }
    return _apply_observation_pipeline(entry, term_cfg, group_history_length)


def _effective_history(group: ObservationGroupCfg, term_cfg: ObservationTermCfg) -> int:
    """Stack depth applied to one term — group level wins, as in mjlab.

    Sparse offsets (``history_steps``) count as their own depth: they are per-term by
    construction, so a group level cannot override them.
    """
    if term_cfg.history_steps:
        return len(term_cfg.history_steps)
    return int(group.history_length or term_cfg.history_length or 0)


def _group_is_fusable(group: ObservationGroupCfg) -> bool:
    """Whether the whole group can become one graph (ADR §4, brief §4b).

    Two things disqualify a group:

    - **A legacy ``*Binding`` term.** It resolves to a hand-written TS class, whose
      body exists only in the browser — there is nothing to trace into the graph.
    - **Per-term history deeper than one frame.** mjlab stacks each term
      *before* concatenating, so the group vector interleaves per-term histories;
      a fused graph emits one concatenation and the runtime's group-level ring
      buffer would stack the whole thing instead, giving step-major order where
      mjlab gives term-major. A depth of 1 is a no-op and stays fusable.

    Anything else — traced bodies, native markers, baked constants — fuses. An
    untraceable term is not handled here: it fails the build either way.
    """
    for term_cfg in group.terms.values():
        if isinstance(term_cfg.func, ObservationBinding):
            return False
        # Sparse offsets disqualify at any length: no fused output can hold a delayed frame.
        if term_cfg.history_steps or _effective_history(group, term_cfg) > 1:
            return False
    return True


def policy_native_sizes(
    data: dict[str, Any], commands: dict[str, CommandTermConfig] | None
) -> dict[str, int]:
    """Widths of the native observation terms, keyed as :func:`_native_size` reads them.

    ``last_action`` and ``generated_commands`` read env-level state, and a trace env
    built for a plain ``add_scene()`` scene has neither an action term nor the
    command (a browser-only ``UiCommand``). A fused graph still needs a fixed
    width, so it comes from the policy's own config: the action count the runtime
    will drive, and the command's value-bearing UI inputs (buttons carry no value —
    see ``UiCommand.getCommand``).
    """
    sizes: dict[str, int] = {}
    num_actions = data.get("policy_num_actions") or len(
        data.get("policy_joint_names") or ()
    )
    if num_actions:
        sizes["prev_action"] = int(num_actions)
    for name, cmd in (commands or {}).items():
        if cmd.ui is None:
            continue
        width = sum(1 for inp in cmd.ui.inputs if not isinstance(inp, ButtonConfig))
        if not width:
            continue
        sizes[f"command:{name}"] = width
    return sizes


def _native_size(
    term_cfg: ObservationTermCfg, native_sizes: dict[str, int]
) -> int | None:
    """Declared width for a native term, or ``None`` if it isn't native."""
    func_name = getattr(term_cfg.func, "__name__", None)
    if func_name == "last_action":
        # The whole vector: a term-scoped `last_action` needs `action_offset` for its slice.
        if term_cfg.params.get("action_name") is not None:
            return None
        return native_sizes.get("prev_action")
    if func_name == "generated_commands":
        return native_sizes.get(f"command:{term_cfg.params['command_name']}")
    return None


def _fused_group_entry(
    group: ObservationGroupCfg,
    env: Any,
    out_dir: Path,
    group_name: str,
    native_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Trace the group as one graph and return the fused config entry."""
    from .compile.tracer import (
        GroupTermSpec,
        slots_json,
        trace_observation_group,
    )

    specs = [
        GroupTermSpec(
            name=name,
            func=term_cfg.func,
            params=_resolved_params(term_cfg.params, env),
            clip=tuple(term_cfg.clip) if term_cfg.clip else None,
            scale=term_cfg.scale,
            native_size=_native_size(term_cfg, native_sizes or {}),
        )
        for name, term_cfg in group.terms.items()
    ]
    export = trace_observation_group(specs, env, name=group_name)
    ref = _onnx_ref("obs", group_name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    entry: dict[str, Any] = {
        "fused": ref,
        "input_slots": slots_json(export),
        "native_inputs": export.native_inputs,
        # Per-term widths in concat order, for the runtime's group layout.
        "layout": export.layout,
        "size": _tensor_width(export.reference_output),
    }
    sensors = _raycast_descriptors(export, env)
    if sensors:
        # Structured sensors only; a builtin one is a `sensordata` window.
        entry["sensors"] = sensors
    return entry


def _mj_element_name(env: Any, obj_type: str, obj_id: int) -> str:
    """Model name of a body/site/geom the sensor's rays are attached to.

    Names travel, not ids: the browser's model is compiled separately, so an id
    from the build env means nothing there.
    """
    mj_model = env.sim.mj_model
    return {"body": mj_model.body, "site": mj_model.site, "geom": mj_model.geom}[
        obj_type
    ](obj_id).name


def raycast_sensor_descriptor(env: Any, sensor_name: str) -> dict[str, Any] | None:
    """Everything the browser needs to reproduce one ``RayCastSensor``'s readings.

    A structured sensor's fields are graph inputs (ADR 0005 §6: state collection is
    native), and unlike a builtin sensor there is no ``sensordata`` window to read —
    the browser has to cast the rays itself. It can: ``mj_ray`` is in the WASM
    build. What it cannot do is re-derive the pattern, so the ray offsets and
    directions are baked here from the live sensor rather than re-implementing
    mjlab's ``GridPatternCfg``/``PinholeCfg``/``RingCfg`` generators — that also
    means a pattern mjswan has never heard of works for free.

    Returns ``None`` if *sensor_name* is not a raycast sensor.
    """
    sensor = env.scene.sensors.get(sensor_name)
    offsets = getattr(sensor, "_local_offsets", None)
    if offsets is None:
        return None
    return {
        "kind": "raycast",
        # [N, 3] each, in the frame's local coordinates.
        "local_offsets": offsets.detach().cpu().tolist(),
        "local_directions": sensor._local_directions.detach().cpu().tolist(),
        "frames": [
            {"type": obj_type, "name": _mj_element_name(env, obj_type, obj_id)}
            for obj_type, obj_id, _ in sensor._frame_infos
        ],
        # "base" | "yaw" | "world" — how the frame's rotation reaches the rays.
        "ray_alignment": sensor.cfg.ray_alignment,
        "max_distance": float(sensor.cfg.max_distance),
        # mjlab excludes each frame's own parent body so a ray cannot self-hit.
        "exclude_parent_body": bool(sensor.cfg.exclude_parent_body),
    }


def _raycast_descriptors(export: Any, env: Any) -> dict[str, Any]:
    """Descriptors for every structured sensor the group's slots name."""
    from .compile.tracer import _SENSOR_NS

    descriptors: dict[str, Any] = {}
    for namespace, name_part in export.input_slots:
        if namespace != _SENSOR_NS or "." not in name_part:
            continue
        sensor_name = name_part.split(".", 1)[0]
        if sensor_name in descriptors:
            continue
        descriptor = raycast_sensor_descriptor(env, sensor_name)
        if descriptor is not None:
            descriptors[sensor_name] = descriptor
    return descriptors


def serialize_observation_group(
    group: ObservationGroupCfg,
    env: Any,
    out_dir: Path,
    group_name: str = "policy",
    native_sizes: dict[str, int] | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Serialize an observation group — one fused graph where possible, else per term.

    Fusion is ADR §4's mandatory-for-v1 optimization: a per-term graph can be a
    single node (three of G1's five are ``Identity``), so the fixed per-``ort.run()``
    cost is the entire expense, and a slot two terms share gets marshalled twice.
    See the companion brief §4b for the measurements.
    """
    from .compile.tracer import ConstantGroup

    if _group_is_fusable(group):
        try:
            return _fused_group_entry(group, env, out_dir, group_name, native_sizes)
        except ConstantGroup:
            # Only knowable by tracing, so not a `_group_is_fusable` static check.
            pass
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
    """Serialize one termination term."""
    from .compile import trace_term
    from .compile.tracer import slots_json

    func = term_cfg.func
    if isinstance(func, TerminationBinding):
        _require_ts_src("Termination", name, func)
        return term_cfg.to_dict()

    try:
        export = trace_term(
            func, _resolved_params(term_cfg.params, env), env, name=name
        )
    except ValueError:
        # No time-varying state read (mjlab's `time_out` counts steps); the threshold rides along.
        entry: dict[str, Any] = {
            "name": name,
            "native": "elapsed_s >= episode_length_s",
            "episode_length_s": float(getattr(env, "max_episode_length_s", 0.0)),
        }
        if term_cfg.time_out:
            entry["time_out"] = True
        return entry

    ref = _onnx_ref("term", name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    entry = {
        "name": name,
        "onnx": ref,
        "input_slots": slots_json(export),
    }
    if term_cfg.time_out:
        entry["time_out"] = True
    return entry


def _native_termination_entry(
    name: str, term_cfg: TerminationTermCfg, env: Any
) -> dict[str, Any]:
    """The `time_out` marker: ADR 0005 §2's one legitimately-native termination.

    It compares env-level step counters rather than entity data, so there is
    nothing to trace. The threshold travels with it — the marker alone names a
    comparison the runtime has no number for.
    """
    entry: dict[str, Any] = {
        "name": name,
        "native": "elapsed_s >= episode_length_s",
        "episode_length_s": float(getattr(env, "max_episode_length_s", 0.0)),
    }
    if term_cfg.time_out:
        entry["time_out"] = True
    return entry


def _is_native_termination(term_cfg: TerminationTermCfg, env: Any) -> bool:
    """Whether a term reads no time-varying state (so it cannot be traced)."""
    from .compile import trace_term
    from .compile.tracer import ConstantTerm

    try:
        trace_term(
            term_cfg.func,
            _resolved_params(term_cfg.params, env),
            env,
            name="probe",
        )
    except ConstantTerm:
        return True
    return False


def serialize_terminations(
    terminations: dict[str, TerminationTermCfg] | None, env: Any, out_dir: Path
) -> dict[str, Any]:
    """Serialize a policy's terminations, fusing the traced ones into one graph.

    Same mechanism and motivation as observation fusion (companion brief §4b),
    with one difference in the output: a bool *lane* per term rather than one
    value, so the manager keeps per-term ``reasons`` and its
    terminated-vs-truncated split.

    Native markers (`time_out`) and legacy `*Binding` terms stay as their own
    entries; the fused graph joins them under ``__fused__``. The gain scales with
    the traced-term count, which is 0–1 for mjlab's locomotion and manipulation
    tasks but 3 for the tracking tasks `examples/mjlab/g1_spinkick` and
    `unitree_rl` use (`anchor_pos`, `anchor_ori`, `ee_body_pos`).
    """
    result: dict[str, Any] = {}
    if not terminations:
        return result

    fusable: dict[str, TerminationTermCfg] = {}
    for name, term_cfg in terminations.items():
        func = term_cfg.func
        if isinstance(func, TerminationBinding):
            _require_ts_src("Termination", name, func)
            result[name] = term_cfg.to_dict()
            continue
        if _is_native_termination(term_cfg, env):
            result[name] = _native_termination_entry(name, term_cfg, env)
            continue
        fusable[name] = term_cfg

    if not fusable:
        return result
    if len(fusable) == 1:
        # Fusing one term buys nothing and costs a wire shape, so don't.
        name, term_cfg = next(iter(fusable.items()))
        entry = serialize_termination(name, term_cfg, env, out_dir)
        if entry is not None:
            result[name] = entry
        return result

    result[FUSED_TERMINATION_KEY] = _fused_termination_entry(
        fusable, env, out_dir, "terminations"
    )
    return result


FUSED_TERMINATION_KEY = "__fused__"
"""Config key the fused termination graph lives under.

Terminations are a name-keyed map, and the fused graph covers several of those
names at once, so it needs a key of its own rather than one term's. The sentinel
cannot collide: mjlab term names come from Python identifiers in a config class.
"""


def _fused_termination_entry(
    terms: dict[str, TerminationTermCfg], env: Any, out_dir: Path, group_name: str
) -> dict[str, Any]:
    from .compile.tracer import (
        GroupTermSpec,
        slots_json,
        trace_termination_group,
    )

    specs = [
        GroupTermSpec(
            name=name, func=cfg.func, params=_resolved_params(cfg.params, env)
        )
        for name, cfg in terms.items()
    ]
    export = trace_termination_group(specs, env, name=group_name)
    ref = _onnx_ref("term", group_name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    return {
        "fused": ref,
        "input_slots": slots_json(export),
        # Lane order is the graph's; `time_out` rides along so the manager can still split it out.
        "lanes": [
            {"name": name, "time_out": bool(terms[name].time_out)}
            for name in export.lanes
        ],
    }


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


_DR_ENTITY_INDEX_ATTR = {
    "geom": "geom_ids",
    "body": "body_ids",
    "site": "site_ids",
}


def _dr_entity_names(env: Any, asset_cfg: Any, entity_type: str) -> list[str] | None:
    """Names of the model elements a startup-DR event perturbs.

    Names, not ids: the browser compiles its own model, so an id from the build env
    means nothing there. ``None`` when the entity type is one this does not know how
    to enumerate — the caller then leaves the event native.
    """
    attr = _DR_ENTITY_INDEX_ATTR.get(entity_type)
    if attr is None:
        return None
    asset = env.scene[asset_cfg.name]
    scoped = getattr(asset_cfg, f"{entity_type}_ids", None)
    all_ids = getattr(asset.indexing, attr)
    ids = [int(i) for i in all_ids.tolist()]
    if scoped is not None and not isinstance(scoped, slice):
        # mjlab's `_get_entity_indices`: positions into the entity's own elements, in cfg order.
        positions = list(scoped) if hasattr(scoped, "__iter__") else [scoped]
        ids = [ids[int(p)] for p in positions]
    accessor = {
        "geom": env.sim.mj_model.geom,
        "body": env.sim.mj_model.body,
        "site": env.sim.mj_model.site,
    }[entity_type]
    return [accessor(i).name for i in ids]


def _dr_arg(func: Any, params: dict[str, Any], key: str) -> Any:
    """A DR keyword as mjlab would see it: the term's value, else *func*'s default.

    The default is read off the wrapper's signature rather than assumed, because
    mjlab's wrappers do not share one: ``geom_friction`` defaults ``operation`` to
    ``"abs"``, ``body_com_offset`` to ``"add"``, ``body_mass`` to ``"scale"``. A
    single hardcoded default would describe an omitted ``operation`` as replacing
    the value when mjlab actually scales it — a silent divergence, and the worse
    one for mass.
    """
    if key in params:
        return params[key]
    param = inspect.signature(func).parameters.get(key)
    return (
        None
        if param is None or param.default is inspect.Parameter.empty
        else param.default
    )


def _dr_name_of(value: Any, fallback: str) -> str:
    """``Operation``/``Distribution`` accept an instance as well as a string."""
    if value is None:
        return fallback
    return str(getattr(value, "name", value))


def _dr_target_axes(
    func: Any, params: dict[str, Any], ranges: Any, default_axes: list[int]
) -> list[int]:
    """mjlab's ``_determine_target_axes`` precedence: explicit, then int keys, then default."""
    axes = _dr_arg(func, params, "axes")
    if axes is not None:
        return [int(a) for a in axes]
    if isinstance(ranges, dict):
        return [int(k) for k in ranges]
    return list(default_axes)


def model_field_dr_descriptor(
    term_cfg: EventTermCfg, env: Any, params: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Describe a startup model-field randomization for the browser, or None.

    These events (`geom_friction`, `body_com_offset`, …) perturb the *model* rather
    than `mjData`, so the `entity_write` tracer has nothing to capture and they look
    to it exactly like an event that did nothing. They also need no graph: the whole
    event is "draw a number per element per axis, combine it with the base value,
    write it back", and the browser can do that itself at startup from the
    orchestrator's seeded PRNG (ADR 0005 §2), so a session still replays.

    Returns ``None`` for anything this cannot describe — an unknown entity type, or
    the string-keyed `ranges` form mjlab resolves by name pattern.
    """
    func = term_cfg.func
    # Resolved params: an unresolved `SceneEntityCfg` widens a scoped event to every geom.
    params = params if params is not None else _resolved_params(term_cfg.params, env)
    field = getattr(func, "_mjswan_dr_field", None) or _DR_FIELD_BY_FUNC.get(
        getattr(func, "__name__", "")
    )
    if field is None:
        return None
    field_name, entity_type, default_axes = field
    ranges = _dr_arg(func, params, "ranges")
    if not isinstance(ranges, (tuple, list, dict)):
        return None
    if isinstance(ranges, dict) and any(isinstance(k, str) for k in ranges):
        # mjlab resolves these by name pattern per element; not described here.
        return None
    asset_cfg = _dr_arg(func, params, "asset_cfg")
    if asset_cfg is None:
        return None
    names = _dr_entity_names(env, asset_cfg, entity_type)
    if names is None:
        return None

    axes = _dr_target_axes(func, params, ranges, default_axes)
    if isinstance(ranges, dict):
        # `_prepare_axis_ranges` drops a range for an axis nobody targets.
        if any(a not in ranges for a in axes):
            return None
        axis_ranges = {a: [float(ranges[a][0]), float(ranges[a][1])] for a in axes}
    else:
        axis_ranges = {a: [float(ranges[0]), float(ranges[1])] for a in axes}

    operation = _dr_name_of(_dr_arg(func, params, "operation"), "abs")
    return {
        "kind": "model_field",
        "field": field_name,
        "entity_type": entity_type,
        "entity_names": names,
        # Axis -> [lo, hi]; only these are written, so events on different axes compose.
        "axis_ranges": axis_ranges,
        "operation": operation,
        "distribution": _dr_name_of(_dr_arg(func, params, "distribution"), "uniform"),
        "shared_random": bool(_dr_arg(func, params, "shared_random")),
        # Which base the browser reads: `add`/`scale` use the compiled default, so they never
        # accumulate across events on one axis.
        "uses_defaults": operation in _DR_OPS_USING_DEFAULTS,
        "set_const": _dr_needs_recompute(func, field_name),
    }


# mjlab's `Operation.uses_defaults`: `abs` reads the live value, `add`/`scale` the default.
_DR_OPS_USING_DEFAULTS = frozenset({"add", "scale"})

# Fallback when a DR func lacks `requires_model_fields`: fields that invalidate constants.
_SET_CONST_FIELDS = frozenset(
    {"body_ipos", "body_mass", "body_inertia", "dof_armature"}
)


def _dr_needs_recompute(func: Any, field_name: str) -> bool:
    """Whether the browser owes an ``mj_setConst`` after writing this field.

    mjlab's `requires_model_fields` decorator records this on the function as a
    `RecomputeLevel`, so read it rather than guessing. Its three non-zero levels
    recompute progressively larger subsets; MuJoCo's C API exposes only the full
    `mj_setConst`, so any level above `none` maps to that — more work than the
    lower levels need, same result.
    """
    recompute = getattr(func, "recompute", None)
    if recompute is not None:
        return int(recompute) > 0
    return field_name in _SET_CONST_FIELDS


_DR_FIELD_BY_FUNC: dict[str, tuple[str, str, list[int]]] = {
    # mjlab's DR helpers wrap `_randomize_model_field` with an entity type and axes that are
    # not introspectable, hence this table. An author's own wrapper can set `_mjswan_dr_field`.
    "geom_friction": ("geom_friction", "geom", [0]),
    "geom_rgba": ("geom_rgba", "geom", [0, 1, 2, 3]),
    "body_com_offset": ("body_ipos", "body", [0, 1, 2]),
    "body_ipos": ("body_ipos", "body", [0, 1, 2]),
    "body_mass": ("body_mass", "body", [0]),
}


_EVENTS_WITH_NOTHING_TO_WRITE: dict[str, str] = {
    # Trace failure is the *expected* outcome for these, for a stated reason. Everything
    # else raises: a silently dropped reset randomization is invisible in both the build
    # output and the browser.
    "randomize_terrain": (
        "it re-draws each env's sub-terrain origin, and the browser has one baked terrain "
        "with one origin"
    ),
    "encoder_bias": (
        "it writes `Entity.data.encoder_bias`, which the runtime applies from the policy "
        "config's `encoder_bias` rather than from an event graph"
    ),
}


def _event_writes_nothing_reason(
    term_cfg: EventTermCfg, env: Any, params: dict[str, Any]
) -> str | None:
    """Why this term's trace legitimately captured no write, or ``None``."""
    func_name = getattr(term_cfg.func, "__name__", "")
    reason = _EVENTS_WITH_NOTHING_TO_WRITE.get(func_name)
    if reason is not None:
        return reason
    if not func_name.startswith("reset_root_state"):
        return None
    # A root-state write cannot move a fixed-base entity — not here, and not in mjlab
    # either. mjlab's manipulation tasks still configure `reset_base` on their arms,
    # leaving `asset_cfg` to the signature default, hence `_dr_arg` and not `params`.
    entity_name = getattr(_dr_arg(term_cfg.func, params, "asset_cfg"), "name", None)
    if entity_name is None:
        return None
    try:
        entity = env.scene[entity_name]
    except (KeyError, TypeError):
        return None
    if getattr(entity, "is_fixed_base", False):
        return f"entity {entity_name!r} is fixed-base, so a root write cannot move it"
    return None


def serialize_event(
    name: str, term_cfg: EventTermCfg, env: Any, out_dir: Path
) -> dict[str, Any] | None:
    """Serialize one event term (any mode). Returns ``None`` if genuinely nothing to emit."""
    from .compile import trace_event_term
    from .compile.tracer import slots_json

    func = term_cfg.func
    if isinstance(func, EventBinding):
        _require_ts_src("Event", name, func)
        return term_cfg.to_dict()

    resolved = _resolved_params(term_cfg.params, env)
    try:
        export = trace_event_term(
            func,
            resolved,
            env,
            name=name,
            mode=term_cfg.mode,
        )
    except ValueError as exc:
        # `entity_write` captures nothing for an `mjModel` write, so describe it instead and
        # let the browser draw from the seeded PRNG at load.
        descriptor = model_field_dr_descriptor(term_cfg, env, resolved)
        if descriptor is not None:
            return {"name": name, "mode": term_cfg.mode, **descriptor}
        nothing_to_write = _event_writes_nothing_reason(term_cfg, env, resolved)
        if nothing_to_write is not None:
            return {
                "name": name,
                "mode": term_cfg.mode,
                "native": True,
                "reason": nothing_to_write,
            }
        raise ValueError(
            f"Event term {name!r} could not be traced: {exc} Emitting it as a no-op "
            "would drop a randomization the task is configured to apply, with nothing "
            "said about it in the browser. Either supply a trace-friendly replacement "
            "via mjswan.register_event(), or write the term as a TS class and point an "
            "EventBinding's `ts_src` at it."
        ) from exc

    ref = _onnx_ref("event", name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    entry: dict[str, Any] = {
        "name": name,
        "mode": term_cfg.mode,
        "onnx": ref,
        "rand_dim": export.rand_dim,
        "rand_ranges": export.rand_ranges,
        "input_slots": slots_json(export),
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


def _serialize_reset_graph(
    name: str, cmd_cfg: CommandTermConfig, env: Any, out_dir: Path
) -> dict[str, Any] | None:
    """Trace a native command's reset-time graph, or ``None`` if it has none.

    The entry shape is deliberately an event entry (:func:`serialize_event`'s), so
    the browser can run it through the same ``OnnxEvent`` handler instead of
    growing a second way to evaluate a graph with ``rand`` and ``entity_write``s.
    """
    pending = cmd_cfg.pending_reset_trace
    if pending is None:
        return None
    from .compile import trace_event_term
    from .compile.tracer import slots_json

    graph_name = f"{name}_reset"
    export = trace_event_term(
        pending.func,
        _resolved_params(pending.params, env),
        env,
        name=graph_name,
        mode="reset",
    )
    ref = _onnx_ref("command", graph_name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    return {
        "name": graph_name,
        "mode": "reset",
        "onnx": ref,
        "rand_dim": export.rand_dim,
        "rand_ranges": export.rand_ranges,
        "input_slots": slots_json(export),
        "write_targets": export.write_targets,
    }


def serialize_command(
    name: str, cmd_cfg: CommandTermConfig, env: Any, out_dir: Path
) -> dict[str, Any]:
    """Serialize one command term, resolving a pending ONNX trace if needed."""
    if cmd_cfg.pending_trace is None:
        reset_graph = _serialize_reset_graph(name, cmd_cfg, env, out_dir)
        if reset_graph is None:
            return cmd_cfg.to_dict()
        # `to_dict` refuses while a trace is pending; the graph is resolved now.
        resolved = replace(cmd_cfg, pending_reset_trace=None)
        return {**resolved.to_dict(), "reset_graph": reset_graph}

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
