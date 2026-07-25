"""OnnxCommand config serialization (ADR 0005 §3, companion brief §3a).

Layer: L1 (pure Python — no mjlab/torch/onnxruntime; builds a CommandExport by
hand). Verifies the config emitted from a traced command carries everything the
runtime needs and validates structurally.
"""

from __future__ import annotations

import json

import pytest

# The mjswan.compile package imports torch at load time (it is the build-time
# tracer); this serialization logic is pure-Python but still needs the package to
# import. Fast (no env build), skipped when the examples extras are absent.
torch = pytest.importorskip("torch")

from mjswan.compile import (  # noqa: E402
    command_config,
    validate_command_config,
    write_command_artifact,
)
from mjswan.compile.tracer import CommandExport  # noqa: E402


def _make_export() -> CommandExport:
    """A velocity-shaped CommandExport (as trace_command_term would return)."""
    return CommandExport(
        name="twist",
        onnx_bytes=b"\x08\x01onnx-graph-bytes",
        state_fields=[
            {"name": "vel_command_b", "shape": [1, 3], "dtype": "float32"},
            {"name": "heading_target", "shape": [1], "dtype": "float32"},
            {"name": "is_heading_env", "shape": [1], "dtype": "bool"},
            {"name": "is_standing_env", "shape": [1], "dtype": "bool"},
        ],
        command_field="vel_command_b",
        input_slots=[("robot", "heading_w")],
        input_names=["robot__heading_w"],
        rand_dim=6,
        output_names=["next_vel_command_b"],
        write_targets=[],
        reference_rand=torch.zeros(6),
    )


def test_command_config_shape():
    export = _make_export()
    cfg = command_config(
        export, onnx_ref="command/twist.onnx", resampling_time_range=(3.0, 8.0)
    )
    assert cfg["type"] == "OnnxCommand"
    assert cfg["name"] == "twist"
    assert cfg["onnx"] == "command/twist.onnx"
    assert cfg["command_field"] == "vel_command_b"
    assert cfg["rand_dim"] == 6
    assert cfg["resampling_time_range"] == [3.0, 8.0]
    # dynamic runtime read threaded as a declared input slot
    assert {"entity": "robot", "field": "heading_w"} in cfg["input_slots"]
    # every state field carries a shape + dtype (brief §3a)
    for sf in cfg["state_fields"]:
        assert set(sf) == {"name", "shape", "dtype"}


def test_command_config_validates():
    export = _make_export()
    cfg = command_config(export, onnx_ref="command/twist.onnx")
    assert validate_command_config(cfg) == []
    # round-trips through JSON unchanged
    assert json.loads(json.dumps(cfg)) == cfg


def test_lifting_command_with_entity_write():
    export = CommandExport(
        name="lift_height",
        onnx_bytes=b"onnx",
        state_fields=[{"name": "target_pos", "shape": [1, 3], "dtype": "float32"}],
        command_field="target_pos",
        input_slots=[],
        input_names=[],
        rand_dim=7,
        output_names=["next_target_pos", "root_pose__pose", "root_velocity__velocity"],
        write_targets=[
            {"kind": "root_pose", "entity": "cube", "fields": ["pose"]},
            {"kind": "root_velocity", "entity": "cube", "fields": ["velocity"]},
        ],
        reference_rand=torch.zeros(7),
    )
    cfg = command_config(export, onnx_ref="command/lift_height.onnx")
    assert validate_command_config(cfg) == []
    kinds = {w["kind"] for w in cfg["write_targets"]}
    assert kinds == {"root_pose", "root_velocity"}


def test_validate_catches_bad_command_field():
    export = _make_export()
    cfg = command_config(export, onnx_ref="command/twist.onnx")
    cfg["command_field"] = "not_a_state_field"
    errors = validate_command_config(cfg)
    assert any("command_field" in e for e in errors)


def test_write_command_artifact(tmp_path):
    export = _make_export()
    cfg = write_command_artifact(export, tmp_path, resampling_time_range=(3.0, 8.0))
    written = tmp_path / "command" / "twist.onnx"
    assert written.read_bytes() == export.onnx_bytes
    assert cfg["onnx"] == "command/twist.onnx"
    assert validate_command_config(cfg) == []
