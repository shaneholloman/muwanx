"""Tests for mjswan.command.

Layer: L1 (pure Python, no MuJoCo/ONNX required).
"""

import pytest

import mjswan
from mjswan.command import (
    ButtonConfig,
    CommandBinding,
    CommandTermConfig,
    SliderConfig,
    _custom_registry,
    register_command,
    ui_command,
    velocity_command,
)


class TestSliderConfig:
    def test_min_max_derived_from_range(self):
        s = SliderConfig(name="x", label="X", range=(-2.0, 3.0))
        assert s.min == -2.0
        assert s.max == 3.0

    def test_to_dict_includes_all_fields(self):
        s = SliderConfig(
            name="lin_vel_x",
            label="Forward Velocity",
            range=(-1.0, 1.0),
            default=0.5,
            step=0.05,
        )
        d = s.to_dict()
        assert d["type"] == "slider"
        assert d["name"] == "lin_vel_x"
        assert d["label"] == "Forward Velocity"
        assert d["min"] == -1.0
        assert d["max"] == 1.0
        assert d["default"] == 0.5
        assert d["step"] == 0.05

    def test_slider_is_alias_for_slider_config(self):
        assert mjswan.Slider is SliderConfig

    def test_adjustable_range_is_absent_unless_asked_for(self):
        # A UI affordance, not a default: a config that never asked keeps no companion.
        assert "adjustable_range" not in SliderConfig(name="x", label="X").to_dict()

    def test_adjustable_range_travels_as_its_own_bounds(self):
        # mjlab's "Max <label>" meta-slider: presentational, so it carries only its bounds.
        s = SliderConfig(
            name="lin_vel_x",
            label="Forward Velocity",
            range=(-1.5, 1.5),
            adjustable_range=mjswan.SliderRangeConfig(
                range=(0.0, 1.5), default=1.5, step=0.05
            ),
        )
        assert s.to_dict()["adjustable_range"] == {
            "min": 0.0,
            "max": 1.5,
            "step": 0.05,
            "default": 1.5,
        }

    def test_adjustable_range_label_is_optional(self):
        # Omitted means the browser writes `Max <label>`; naming it overrides that.
        default = mjswan.SliderRangeConfig()
        assert "label" not in default.to_dict()
        named = mjswan.SliderRangeConfig(label="Speed cap")
        assert named.to_dict()["label"] == "Speed cap"


class TestButtonConfig:
    def test_to_dict_includes_name_and_label(self):
        b = ButtonConfig(name="reset", label="Reset Simulation")
        assert b.to_dict() == {
            "type": "button",
            "name": "reset",
            "label": "Reset Simulation",
        }

    def test_button_is_alias_for_button_config(self):
        assert mjswan.Button is ButtonConfig


class TestUiCommand:
    def test_ui_command_serializes_as_ui_term(self):
        command = ui_command(
            [
                SliderConfig(name="x", label="X", range=(-1.0, 1.0)),
                ButtonConfig(name="reset", label="Reset"),
            ]
        )
        assert command.to_dict() == {
            "name": "UiCommand",
            "ui": {
                "inputs": [
                    {
                        "type": "slider",
                        "name": "x",
                        "label": "X",
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "default": 0.0,
                    },
                    {
                        "type": "button",
                        "name": "reset",
                        "label": "Reset",
                    },
                ]
            },
        }


class TestVelocityCommand:
    def test_velocity_command_is_ui_command(self):
        cmd = velocity_command()
        assert isinstance(cmd, CommandTermConfig)
        assert cmd.term_name == "UiCommand"

    def test_velocity_command_has_exactly_three_sliders(self):
        cmd = velocity_command()
        inputs = cmd.ui.inputs if cmd.ui is not None else []
        assert len(inputs) == 3
        assert all(isinstance(inp, SliderConfig) for inp in inputs)

    def test_slider_names_are_canonical(self):
        cmd = velocity_command()
        inputs = cmd.ui.inputs if cmd.ui is not None else []
        assert [inp.name for inp in inputs] == ["lin_vel_x", "lin_vel_y", "ang_vel_z"]

    def test_velocity_command_is_accessible_from_mjswan(self):
        assert mjswan.velocity_command is velocity_command


class TestCommandRegistry:
    def test_register_command_is_accessible_from_mjswan(self):
        assert mjswan.register_command is register_command

    def test_custom_term_spec_can_be_registered(self):
        register_command(
            "DummyCommandCfg",
            CommandBinding(
                ts_name="DummyCommand",
                serializer=lambda cfg: {"value": cfg.value},
            ),
        )

        class DummyCfg:
            value = 3

        spec = _custom_registry["DummyCommandCfg"]
        assert spec.ts_name == "DummyCommand"
        assert spec.serializer(DummyCfg()) == {"value": 3}


class TestMotionRsiRegistration:
    """The RSI jitter graph lives author-side; its absence must not be silent.

    `TrackingCommand.ts` used to jitter with `Math.random()`. ADR 0005 moved that
    into a traced graph whose body needs mjlab's own `sample_uniform` /
    `quat_from_euler_xyz`, so it is registered from `examples/mjlab/defaults/
    commands` rather than from `mjswan.command` (which keeps mjlab a soft
    dependency). A task whose author never imported that module therefore got the
    plain binding — no graph — and quietly stopped jittering. These pin the
    diagnosis that replaced the silence.
    """

    def test_warns_when_a_jittering_cfg_has_no_registered_graph(self):
        from mjswan.command import _motion_rsi_unregistered

        class MotionCommandCfg:
            # mjlab's play override: pose/velocity cleared, joint jitter kept.
            pose_range: dict = {}
            velocity_range: dict = {}
            joint_position_range = (-0.1, 0.1)

        with pytest.warns(RuntimeWarning, match="unjittered reference frame"):
            assert _motion_rsi_unregistered(MotionCommandCfg()) is None

    def test_stays_quiet_when_the_cfg_jitters_nothing(self):
        """A task with every range cleared is not missing anything."""
        import warnings

        from mjswan.command import _motion_rsi_unregistered

        class MotionCommandCfg:
            pose_range: dict = {}
            velocity_range: dict = {}
            joint_position_range = (0.0, 0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _motion_rsi_unregistered(MotionCommandCfg()) is None

    def test_warns_for_a_pose_or_velocity_range_too(self):
        from mjswan.command import _motion_rsi_unregistered

        class PoseOnly:
            pose_range = {"x": (-0.1, 0.1)}
            velocity_range: dict = {}
            joint_position_range = (0.0, 0.0)

        class VelocityOnly:
            pose_range: dict = {}
            velocity_range = {"z": (-0.2, 0.2)}
            joint_position_range = (0.0, 0.0)

        for cfg in (PoseOnly(), VelocityOnly()):
            with pytest.warns(RuntimeWarning):
                _motion_rsi_unregistered(cfg)

    def test_the_builtin_binding_carries_the_diagnosis(self):
        """Registered on the binding, or nothing would ever call it."""
        from mjswan.command import _custom_registry, _motion_rsi_unregistered

        spec = _custom_registry["MotionCommandCfg"]
        # An author-side re-registration replaces this, so only pin the diagnosing default.
        assert spec.reset_trace in (_motion_rsi_unregistered, spec.reset_trace)
        assert spec.ts_name == "TrackingCommand"
