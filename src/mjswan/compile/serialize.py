"""Serialize traced terms into the ONNX artifact bundle (ADR 0005 §1, §6).

Turns a :class:`~mjswan.compile.tracer.CommandExport` into the ``OnnxCommand``
config entry the runtime consumes and writes the ``.onnx`` graph beside it. Per
the companion brief §1, this reuses the existing ``config.json``/``policy.json``
contract — the config entry here is what replaces a command's DSL/`UiCommand`
mapping in ``policy.json``.

A single generic ``OnnxCommand`` runtime handler interprets every command
(velocity, lifting, …) from this data — there is no engine-side class per command
(brief §3). The config declares everything the handler needs to allocate state,
supply ``rand``, thread dynamic reads, and apply any ``entity_write``:

- ``state_fields``  — each with **shape + dtype + init** so the handler can build the
  ONNX I/O tensors and persist state across frames (brief §3a).
- ``command_field`` — which state field is the command value.
- ``rand_dim``      — how many seeded PRNG draws to feed as ``rand``.
- ``input_slots``   — time-varying runtime reads threaded as graph inputs.
- ``write_targets`` — any ``entity_write`` (cube/root pose+velocity) the graph emits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tracer import CommandExport, slots_json

# Authoritative JSON Schema for one OnnxCommand config entry. The TS runtime
# validates the manifest against this at load time (brief §6); the Python emitter
# self-checks against the lighter `validate_command_config` below.
#
# "name" is the registry key CommandManager looks up a class by — always the
# literal "OnnxCommand" here, matching every other *_command() factory's wire
# convention in command.py, not the term's own identity (that is the outer dict
# key in PolicyConfig.commands, e.g. commands={"twist": ...}). "term_id" carries
# the traced term's own name for diagnostics only.
COMMAND_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OnnxCommand",
    "type": "object",
    "required": [
        "name",
        "onnx",
        "command_field",
        "rand_dim",
        "state_fields",
        "input_slots",
        "write_targets",
    ],
    "additionalProperties": False,
    "properties": {
        "name": {"const": "OnnxCommand"},
        "term_id": {"type": "string"},
        "onnx": {"type": "string"},
        "command_field": {"type": "string"},
        "rand_dim": {"type": "integer", "minimum": 0},
        "state_fields": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "shape", "dtype", "init"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "shape": {"type": "array", "items": {"type": "integer"}},
                    "dtype": {"type": "string"},
                    # Flattened initial value (ADR 0005 §3), so the runtime starts
                    # the term where the build found it rather than at zero.
                    "init": {"type": "array", "items": {"type": ["number", "boolean"]}},
                },
            },
        },
        # Three slot shapes (mjswan.compile.tracer.slot_to_json): an ``Entity.data``
        # read carries entity+field; a whole-sensor read carries sensor; another
        # command term's state carries command+field. All carry ``input`` (the graph
        # input name to feed the value as) and ``shape`` (the traced rank, which the
        # runtime cannot recover from a flat value array).
        "input_slots": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["input"],
                "additionalProperties": False,
                "oneOf": [
                    {"required": ["entity", "field"]},
                    {"required": ["sensor"]},
                    {"required": ["command", "field"]},
                ],
                "properties": {
                    "entity": {"type": ["string", "null"]},
                    "field": {"type": "string"},
                    "sensor": {"type": "string"},
                    "command": {"type": "string"},
                    "input": {"type": "string"},
                    "shape": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
        "write_targets": {"type": "array", "items": {"type": "object"}},
        "resampling_time_range": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "debug_vis": {"type": "boolean"},
        "ui": {"type": "object"},
        "viz": {
            "type": "object",
            "required": ["field", "shape", "radius", "color"],
            "additionalProperties": False,
            "properties": {
                "field": {"type": "string"},
                "shape": {"const": "sphere"},
                "radius": {"type": "number"},
                "color": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
            },
        },
    },
}


def command_config(
    export: CommandExport,
    *,
    onnx_ref: str,
    resampling_time_range: tuple[float, float] | None = None,
    debug_vis: bool = False,
    ui: dict[str, Any] | None = None,
    viz: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``OnnxCommand`` config entry for ``policy.json`` from a trace.

    Follows the same wire convention as every other command term in this codebase
    (``CommandTermConfig.to_dict()``): ``"name"`` is the **registry key** the
    TS-side ``CommandManager`` looks up a class by (here, the constant
    ``"OnnxCommand"`` — the one generic handler, ADR 0005 §3), not the term's own
    identity. The term's own id is the *outer* dict key the author chooses when
    placing this entry into ``PolicyConfig.commands`` (e.g.
    ``commands={"twist": ...}``) — mirrored by every existing ``*_command()``
    factory in ``command.py``. ``export.name`` is kept only as ``term_id`` for
    diagnostics and is not consumed by ``CommandManager``.

    Args:
        export: The traced command (:func:`~mjswan.compile.tracer.trace_command_term`).
        onnx_ref: Bundle-relative path to the written ``.onnx`` graph.
        resampling_time_range: The command's ``[min, max]`` resample seconds
            (from ``cfg.resampling_time_range``) — the native timer uses it.
        debug_vis: Mirror of ``cfg.debug_vis``.
        ui: Optional authored UI descriptor (checkbox/sliders/button, brief §3a).
            Not derivable from the trace; the task author supplies it.
        viz: Optional generic debug-vis descriptor: a 3D-position ``state_fields``
            entry rendered as a sphere marker (``OnnxCommand.updateDebugVisuals``),
            visible only while ``debug_vis`` is true. Replaces a per-command
            hand-written TS class for this — e.g. ``LiftingCommand``'s target
            marker. Not derivable from the trace; the task author supplies it
            (e.g. from ``cfg.viz.target_color``).
    """
    cfg: dict[str, Any] = {
        "name": "OnnxCommand",
        "term_id": export.name,
        "onnx": onnx_ref,
        "command_field": export.command_field,
        "rand_dim": export.rand_dim,
        "state_fields": export.state_fields,
        "input_slots": slots_json(export),
        "write_targets": export.write_targets,
        "debug_vis": bool(debug_vis),
    }
    if resampling_time_range is not None:
        cfg["resampling_time_range"] = [float(v) for v in resampling_time_range]
    if ui is not None:
        cfg["ui"] = ui
    if viz is not None:
        cfg["viz"] = viz
    return cfg


def write_command_artifact(
    export: CommandExport,
    out_dir: str | Path,
    *,
    resampling_time_range: tuple[float, float] | None = None,
    debug_vis: bool = False,
    ui: dict[str, Any] | None = None,
    viz: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``<out_dir>/command/<name>.onnx`` and return its config entry."""
    onnx_ref = f"command/{export.name}.onnx"
    path = Path(out_dir) / onnx_ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(export.onnx_bytes)
    return command_config(
        export,
        onnx_ref=onnx_ref,
        resampling_time_range=resampling_time_range,
        debug_vis=debug_vis,
        ui=ui,
        viz=viz,
    )


def validate_command_config(cfg: dict[str, Any]) -> list[str]:
    """Lightweight structural check of an OnnxCommand config (no jsonschema dep).

    Returns a list of human-readable errors (empty if valid). The authoritative
    contract is :data:`COMMAND_JSON_SCHEMA`, enforced browser-side at load time;
    this catches emitter mistakes early in the Python build.
    """
    errors: list[str] = []
    required: dict[str, type | tuple[type, ...]] = {
        "name": str,
        "onnx": str,
        "command_field": str,
        "rand_dim": int,
        "state_fields": list,
        "input_slots": list,
        "write_targets": list,
    }
    for key, typ in required.items():
        if key not in cfg:
            errors.append(f"missing '{key}'")
        elif not isinstance(cfg[key], typ):
            errors.append(f"'{key}' must be {getattr(typ, '__name__', typ)}")

    if cfg.get("name") != "OnnxCommand":
        errors.append("'name' must be 'OnnxCommand' (the registry key)")

    names: set[str] = set()
    for sf in cfg.get("state_fields", []):
        if not isinstance(sf, dict) or not {"name", "shape", "dtype", "init"} <= set(
            sf
        ):
            errors.append(f"state_field must have name/shape/dtype/init: {sf!r}")
            continue
        expected = 1
        for dim in sf["shape"]:
            expected *= int(dim)
        if len(sf["init"]) != expected:
            errors.append(
                f"state_field {sf['name']!r}: init has {len(sf['init'])} values, "
                f"shape {sf['shape']} needs {expected}"
            )
        names.add(sf["name"])
    if cfg.get("command_field") and cfg["command_field"] not in names:
        errors.append(
            f"command_field {cfg['command_field']!r} is not a declared state field"
        )

    for slot in cfg.get("input_slots", []):
        if not isinstance(slot, dict) or "input" not in slot:
            errors.append(f"input_slot must have 'input': {slot!r}")
        elif "sensor" not in slot and "field" not in slot:
            errors.append(f"input_slot must have 'sensor' or 'field': {slot!r}")

    rtr = cfg.get("resampling_time_range")
    if rtr is not None and (not isinstance(rtr, list) or len(rtr) != 2):
        errors.append("resampling_time_range must be [min, max]")

    viz = cfg.get("viz")
    if viz is not None:
        if not isinstance(viz, dict) or not {
            "field",
            "shape",
            "radius",
            "color",
        } <= set(viz):
            errors.append(f"viz must have field/shape/radius/color: {viz!r}")
        elif viz["field"] not in names:
            errors.append(f"viz field {viz['field']!r} is not a declared state field")

    return errors
