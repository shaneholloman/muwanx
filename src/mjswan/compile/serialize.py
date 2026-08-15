"""Serialize a traced command into its ``policy.json`` entry and ``.onnx`` graph.

One generic ``OnnxCommand`` runtime handler interprets every command from this data,
so the entry has to declare everything it needs to allocate state, supply ``rand``,
thread dynamic reads, and apply any ``entity_write``. The shape is defined by
:func:`command_config` below and consumed by ``core/command/OnnxCommand.ts``; both
sides are covered by tests, so there is no third restatement of it here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tracer import CommandExport, slots_json


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
