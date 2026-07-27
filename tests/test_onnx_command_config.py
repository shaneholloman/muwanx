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
from mjswan.compile.tracer import (  # noqa: E402
    _COMMAND_NS,
    _SENSOR_NS,
    CommandExport,
    _is_dynamic_field,
    slot_to_json,
)


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
    # "name" is the CommandManager registry key (always "OnnxCommand" — the one
    # generic handler); the term's own id ("twist") is the caller's dict key in
    # PolicyConfig.commands, kept here only as "term_id" for diagnostics.
    assert cfg["name"] == "OnnxCommand"
    assert cfg["term_id"] == "twist"
    assert cfg["onnx"] == "command/twist.onnx"
    assert cfg["command_field"] == "vel_command_b"
    assert cfg["rand_dim"] == 6
    assert cfg["resampling_time_range"] == [3.0, 8.0]
    # dynamic runtime read threaded as a declared input slot; `input` names the
    # graph input to feed it as, so the runtime never re-derives it
    assert {
        "entity": "robot",
        "field": "heading_w",
        "input": "robot__heading_w",
    } in cfg["input_slots"]
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


def test_slot_to_json_entity_data():
    assert slot_to_json(("robot", "joint_pos")) == {
        "entity": "robot",
        "field": "joint_pos",
        "input": "robot__joint_pos",
    }


def test_slot_to_json_sensor():
    # A whole-sensor read (mjlab's builtin_sensor) is its own slot shape, and the
    # MJCF path in the name is folded to an identifier for the graph input.
    assert slot_to_json((_SENSOR_NS, "robot/imu_lin_vel")) == {
        "sensor": "robot/imu_lin_vel",
        "input": "sensor__robot_imu_lin_vel",
    }


def test_slot_to_json_command_state():
    # A read of another command's state (mjlab's object_to_goal_distance).
    assert slot_to_json((_COMMAND_NS, "lift_height.target_pos")) == {
        "command": "lift_height",
        "field": "target_pos",
        "input": "command__lift_height_target_pos",
    }


def test_unknown_data_fields_default_to_dynamic():
    # Baking a field that actually varies is silent corruption, so only the
    # model-derived constants are listed and anything else errs toward dynamic.
    # (`site_pos_w` was silently frozen while the allowlist ran the other way.)
    assert _is_dynamic_field("site_pos_w")
    assert _is_dynamic_field("some_future_mjlab_field")
    assert not _is_dynamic_field("default_joint_pos")
    assert not _is_dynamic_field("soft_joint_pos_limits")


def test_command_config_accepts_a_sensor_slot():
    export = _make_export()
    export.input_slots = [(_SENSOR_NS, "robot/imu_ang_vel")]
    cfg = command_config(export, onnx_ref="command/twist.onnx")
    assert cfg["input_slots"] == [
        {"sensor": "robot/imu_ang_vel", "input": "sensor__robot_imu_ang_vel"}
    ]
    assert validate_command_config(cfg) == []


def test_validate_catches_slot_without_input_name():
    export = _make_export()
    cfg = command_config(export, onnx_ref="command/twist.onnx")
    del cfg["input_slots"][0]["input"]
    assert any("input" in e for e in validate_command_config(cfg))


def test_write_command_artifact(tmp_path):
    export = _make_export()
    cfg = write_command_artifact(export, tmp_path, resampling_time_range=(3.0, 8.0))
    written = tmp_path / "command" / "twist.onnx"
    assert written.read_bytes() == export.onnx_bytes
    assert cfg["onnx"] == "command/twist.onnx"
    assert validate_command_config(cfg) == []


# ---------------------------------------------------------------------------
# A native command's traced reset graph (ADR 0005 §3): `MotionCommand`'s
# reference-state-initialization jitter. The motion player stays native — a clip
# lookup is not term math — while the `sample_uniform` around it is traced, so the
# browser needs no hand-written randomness for it.
# ---------------------------------------------------------------------------


def _write_capture_env():
    """Minimal env satisfying the event tracer: `scene[name].data.<field>` + writes."""

    class _Data:
        def __init__(self, **fields):
            for key, value in fields.items():
                setattr(self, key, value)

    class _Entity:
        def __init__(self, data):
            self.data = data

        def write_joint_state_to_sim(self, position, velocity, **_):
            pass

    class _Scene:
        def __init__(self, entities):
            self._entities = entities
            self.env_origins = torch.zeros((1, 3))

        def __getitem__(self, name):
            return self._entities[name]

    class _Env:
        def __init__(self, entities):
            self.scene = _Scene(entities)

    data = _Data(
        joint_pos=torch.tensor([[0.1, 0.2]]),
        joint_vel=torch.tensor([[0.0, 0.0]]),
    )
    return _Env({"robot": _Entity(data)})


class _AssetCfg:
    """Stands in for mjlab's SceneEntityCfg (only `.name` is read here)."""

    def __init__(self, name):
        self.name = name


def test_native_command_emits_a_traced_reset_graph(tmp_path):
    pytest.importorskip("mjlab")
    from rsi_body_fixture import rsi_joint_offset

    from mjswan._onnx_build import serialize_command
    from mjswan.command import CommandTermConfig, PendingResetTrace

    cfg = CommandTermConfig(
        term_name="TrackingCommand",
        params={"sampling_mode": "start"},
        pending_reset_trace=PendingResetTrace(
            func=rsi_joint_offset,
            params={"asset_cfg": _AssetCfg("robot"), "offset": 0.1},
        ),
    )
    entry = serialize_command("motion", cfg, _write_capture_env(), tmp_path)

    # The native term's own params survive untouched...
    assert entry["name"] == "TrackingCommand"
    assert entry["sampling_mode"] == "start"
    # ...and the graph rides alongside them in exactly the shape `OnnxEvent`
    # consumes, so the runtime needs no second way to evaluate a graph that draws
    # `rand` and emits entity writes.
    graph = entry["reset_graph"]
    assert graph["mode"] == "reset"
    assert graph["onnx"] == "command/motion_reset.onnx"
    assert graph["rand_dim"] == 2  # one draw per joint
    assert [t["kind"] for t in graph["write_targets"]] == ["joint_state"]
    assert [s["field"] for s in graph["input_slots"]] == ["joint_pos", "joint_vel"]
    assert (tmp_path / graph["onnx"]).exists()


def test_command_without_a_reset_trace_is_unchanged(tmp_path):
    from mjswan._onnx_build import serialize_command
    from mjswan.command import CommandTermConfig

    cfg = CommandTermConfig(
        term_name="TrackingCommand", params={"sampling_mode": "start"}
    )
    # No env is touched at all when there is nothing to trace.
    entry = serialize_command("motion", cfg, object(), tmp_path)
    assert entry == {"name": "TrackingCommand", "sampling_mode": "start"}


# ---------------------------------------------------------------------------
# An observation term the tracer cannot follow must fail the build, not degrade.
# Both degradations shipped a silently-wrong policy: dropping the term shortens
# the vector the network was trained on, and baking a time-varying term freezes an
# input. mjlab's `height_scan` hit the second one on both Velocity-Rough tasks —
# 187 frozen terrain heights, fed forever, with nothing in the output saying so.
# ---------------------------------------------------------------------------


def _opaque_state_env():
    """An env whose only readable state is not a tensor, so no slot can carry it."""

    class _Data:
        def __init__(self):
            # Real state the term branches on, but nothing a graph input can hold.
            self.contact_mode = "soft"

    class _Entity:
        def __init__(self):
            self.data = _Data()

    class _Scene:
        def __init__(self):
            self.sensors = {}
            self._entities = {"robot": _Entity()}

        def __getitem__(self, name):
            return self._entities[name]

    class _Env:
        def __init__(self):
            self.scene = _Scene()

    return _Env()


def _reads_opaque_state(env, *, entity_name="robot"):
    """A term whose only env read yields no tensor — the untraceable shape.

    mjlab's `height_scan` used to land here (its `RayCastSensor.data` is a struct
    of ray hits); the tracer now follows that per field, so this fixture uses a
    non-tensor field instead. The failure mode being guarded is unchanged: state
    was read, none of it can become a graph input, and baking the result would
    freeze whatever it varies with.
    """
    mode = env.scene[entity_name].data.contact_mode
    return torch.full((1, 3), 1.0 if mode == "soft" else 2.0)


def _reads_nothing(env, *, width=3):
    """A genuine constant — a fixed-size padding term with no env dependency."""
    del env
    return torch.zeros((1, width))


def test_untraceable_observation_fails_the_build():
    from mjswan.compile.tracer import UntraceableTerm, trace_term

    with pytest.raises(UntraceableTerm) as excinfo:
        trace_term(_reads_opaque_state, {}, _opaque_state_env(), name="contact_obs")
    # The message has to name what it could not follow, or nobody can act on it.
    assert "contact_obs" in str(excinfo.value)
    assert "robot.contact_mode" in str(excinfo.value)
    assert excinfo.value.touched == ["robot.contact_mode"]


def test_term_reading_nothing_is_a_constant_not_untraceable():
    from mjswan.compile.tracer import ConstantTerm, UntraceableTerm, trace_term

    with pytest.raises(ConstantTerm) as excinfo:
        trace_term(_reads_nothing, {}, _opaque_state_env(), name="padding")
    # A `ConstantTerm` is safe to bake; an `UntraceableTerm` is not. They are
    # indistinguishable from "no graph inputs" alone, hence two types.
    assert not isinstance(excinfo.value, UntraceableTerm)


def test_serializer_bakes_a_constant_but_refuses_an_untraceable_term(tmp_path):
    from mjswan._onnx_build import serialize_observation_term
    from mjswan.compile.tracer import UntraceableTerm
    from mjswan.managers.observation_manager import ObservationTermCfg

    env = _opaque_state_env()
    baked = serialize_observation_term(
        "padding", ObservationTermCfg(func=_reads_nothing), env, tmp_path, None
    )
    assert baked["native"] == "constant"
    assert baked["size"] == 3

    with pytest.raises(UntraceableTerm):
        serialize_observation_term(
            "contact_obs",
            ObservationTermCfg(func=_reads_opaque_state),
            env,
            tmp_path,
            None,
        )


def test_unsupported_observation_binding_fails_rather_than_dropping():
    from mjswan._onnx_build import serialize_observation_term
    from mjswan.envs.mdp.observations import ObservationBinding
    from mjswan.managers.observation_manager import ObservationTermCfg

    term = ObservationTermCfg(
        func=ObservationBinding(ts_name="", unsupported_reason="no RayCastSensor.")
    )
    with pytest.raises(ValueError, match="shorter observation vector"):
        serialize_observation_term("height_scan", term, object(), None, None)


def test_structured_sensor_fields_become_one_slot_each():
    """mjlab's `RayCastSensor` traces per field, not as one opaque blob.

    A height scan reads `distances` / `frame_pos_w` / `hit_pos_w` off a sensor whose
    `.data` is a struct. Logging the struct wholesale made the term look
    untraceable, which is what silently froze `height_scan` on both Velocity-Rough
    tasks. Each field is its own slot now, so the arithmetic traces and the runtime
    is told exactly which readings to supply.
    """
    from mjswan.compile.tracer import slot_to_json, trace_term

    class _RayData:
        def __init__(self):
            self.distances = torch.tensor([[1.0, 2.0]])
            self.hit_pos_w = torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]]])

    class _Sensor:
        def __init__(self):
            self.data = _RayData()

    class _Scene:
        def __init__(self):
            self.sensors = {"terrain_scan": _Sensor()}

        def __getitem__(self, name):
            return self.sensors[name]

    class _Env:
        def __init__(self):
            self.scene = _Scene()

    def height_scan(env, *, sensor_name="terrain_scan"):
        data = env.scene[sensor_name].data
        heights = -data.hit_pos_w[..., 2]
        return torch.where(data.distances < 0, torch.zeros_like(heights), heights)

    export = trace_term(height_scan, {}, _Env(), name="height_scan")
    assert [slot_to_json(k) for k in export.input_slots] == [
        {
            "sensor": "terrain_scan",
            "input": "sensor__terrain_scan_distances",
            "field": "distances",
        },
        {
            "sensor": "terrain_scan",
            "input": "sensor__terrain_scan_hit_pos_w",
            "field": "hit_pos_w",
        },
    ]
    # Only the fields the term touched — `normals_w` and the rest stay out.
    assert export.reference_output.shape == (1, 2)
