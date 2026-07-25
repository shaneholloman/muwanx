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

- ``state_fields``  — each with **shape + dtype** so the handler can build the
  ONNX I/O tensors and persist state across frames (brief §3a).
- ``command_field`` — which state field is the command value.
- ``rand_dim``      — how many seeded PRNG draws to feed as ``rand``.
- ``input_slots``   — time-varying runtime reads threaded as graph inputs.
- ``write_targets`` — any ``entity_write`` (cube/root pose+velocity) the graph emits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tracer import CommandExport

# Authoritative JSON Schema for one OnnxCommand config entry. The TS runtime
# validates the manifest against this at load time (brief §6); the Python emitter
# self-checks against the lighter `validate_command_config` below.
COMMAND_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OnnxCommand",
    "type": "object",
    "required": [
        "name",
        "type",
        "onnx",
        "command_field",
        "rand_dim",
        "state_fields",
        "input_slots",
        "write_targets",
    ],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "type": {"const": "OnnxCommand"},
        "onnx": {"type": "string"},
        "command_field": {"type": "string"},
        "rand_dim": {"type": "integer", "minimum": 0},
        "state_fields": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "shape", "dtype"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "shape": {"type": "array", "items": {"type": "integer"}},
                    "dtype": {"type": "string"},
                },
            },
        },
        "input_slots": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["entity", "field"],
                "additionalProperties": False,
                "properties": {
                    "entity": {"type": ["string", "null"]},
                    "field": {"type": "string"},
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
    },
}


def command_config(
    export: CommandExport,
    *,
    onnx_ref: str,
    resampling_time_range: tuple[float, float] | None = None,
    debug_vis: bool = False,
    ui: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``OnnxCommand`` config entry for ``policy.json`` from a trace.

    Args:
        export: The traced command (:func:`~mjswan.compile.tracer.trace_command_term`).
        onnx_ref: Bundle-relative path to the written ``.onnx`` graph.
        resampling_time_range: The command's ``[min, max]`` resample seconds
            (from ``cfg.resampling_time_range``) — the native timer uses it.
        debug_vis: Mirror of ``cfg.debug_vis``.
        ui: Optional authored UI descriptor (checkbox/sliders/button, brief §3a).
            Not derivable from the trace; the task author supplies it.
    """
    cfg: dict[str, Any] = {
        "name": export.name,
        "type": "OnnxCommand",
        "onnx": onnx_ref,
        "command_field": export.command_field,
        "rand_dim": export.rand_dim,
        "state_fields": export.state_fields,
        "input_slots": [{"entity": e, "field": f} for e, f in export.input_slots],
        "write_targets": export.write_targets,
        "debug_vis": bool(debug_vis),
    }
    if resampling_time_range is not None:
        cfg["resampling_time_range"] = [float(v) for v in resampling_time_range]
    if ui is not None:
        cfg["ui"] = ui
    return cfg


def write_command_artifact(
    export: CommandExport,
    out_dir: str | Path,
    *,
    resampling_time_range: tuple[float, float] | None = None,
    debug_vis: bool = False,
    ui: dict[str, Any] | None = None,
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
        "type": str,
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

    if cfg.get("type") != "OnnxCommand":
        errors.append("'type' must be 'OnnxCommand'")

    names: set[str] = set()
    for sf in cfg.get("state_fields", []):
        if not isinstance(sf, dict) or not {"name", "shape", "dtype"} <= set(sf):
            errors.append(f"state_field must have name/shape/dtype: {sf!r}")
            continue
        names.add(sf["name"])
    if cfg.get("command_field") and cfg["command_field"] not in names:
        errors.append(
            f"command_field {cfg['command_field']!r} is not a declared state field"
        )

    for slot in cfg.get("input_slots", []):
        if not isinstance(slot, dict) or "field" not in slot:
            errors.append(f"input_slot must have 'field': {slot!r}")

    rtr = cfg.get("resampling_time_range")
    if rtr is not None and (not isinstance(rtr, list) or len(rtr) != 2):
        errors.append("resampling_time_range must be [min, max]")

    return errors
