"""Serialize a traced command into its ``policy.json`` entry and ``.onnx`` graph.

One generic ``OnnxCommand`` runtime handler interprets every command from this data,
so the entry has to declare everything it needs to allocate state, supply ``rand``,
thread dynamic reads, and apply any ``entity_write`` — see
:data:`COMMAND_JSON_SCHEMA`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tracer import CommandExport, slots_json

# Authoritative schema for one OnnxCommand config entry; the TS runtime validates
# against it at load time. "name" is the registry key CommandManager resolves a class
# by, so it is always the literal "OnnxCommand"; "term_id" is the traced term's name.
COMMAND_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OnnxCommand",
    "type": "object",
    "required": [
        "name",
        "onnx",
        "command_field",
        "rand_dim",
        "rand_ranges",
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
        # One [low, high] per `rand` element, in draw order — the graph has none.
        "rand_ranges": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
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
                    # So the runtime starts the term where the build found it.
                    "init": {"type": "array", "items": {"type": ["number", "boolean"]}},
                },
            },
        },
        # Slot shapes per `tracer.slot_to_json`; all carry ``input`` and ``shape``.
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

    The term's own id is the outer key the author gives this entry in
    ``PolicyConfig.commands``; ``term_id`` here is only for diagnostics.

    ``ui`` (control-panel inputs) and ``viz`` (a ``state_fields`` entry to render as a
    sphere marker while ``debug_vis`` is on) are not derivable from the trace — the
    task author supplies them.
    """
    cfg: dict[str, Any] = {
        "name": "OnnxCommand",
        "term_id": export.name,
        "onnx": onnx_ref,
        "command_field": export.command_field,
        "rand_dim": export.rand_dim,
        "rand_ranges": export.rand_ranges,
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
    """Structural check of an OnnxCommand config, returning human-readable errors.

    Catches emitter mistakes early; :data:`COMMAND_JSON_SCHEMA` remains the
    authoritative contract, enforced browser-side at load time.
    """
    errors: list[str] = []
    required: dict[str, type | tuple[type, ...]] = {
        "name": str,
        "onnx": str,
        "command_field": str,
        "rand_dim": int,
        "rand_ranges": list,
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

    # One [low, high] per rand element, or the runtime silently draws [0, 1) instead.
    ranges = cfg.get("rand_ranges")
    if isinstance(ranges, list) and len(ranges) != cfg.get("rand_dim", 0):
        errors.append(
            f"rand_ranges has {len(ranges)} entries, rand_dim is {cfg.get('rand_dim')}"
        )

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
