"""Tests for mjswan.adapters.mjlab_adapter — mjlab type conversion.

Layer: L1 (pure Python, no MuJoCo/ONNX/mjlab required).

These tests simulate mjlab types by creating lightweight mock classes
with the same attributes that the adapter inspects, placed in a fake
``mjlab.*`` module path.
"""

from __future__ import annotations

from typing import Any

import pytest

from mjswan.adapters.mjlab_adapter import (
    adapt_actions,
    adapt_commands,
    adapt_observations,
    adapt_terminations,
)
from mjswan.envs.mdp.observations import ObservationBinding
from mjswan.envs.mdp.terminations import TerminationBinding
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg

# ---------------------------------------------------------------------------
# Fake mjlab types — classes whose __module__ starts with "mjlab"
# ---------------------------------------------------------------------------


def _make_mjlab_class(class_name: str, **defaults: Any) -> type:
    """Create a simple dataclass-like class that appears to come from mjlab."""

    class Cls:
        def __init__(self, **kwargs: Any):
            for k, v in {**defaults, **kwargs}.items():
                setattr(self, k, v)

    Cls.__name__ = class_name
    Cls.__qualname__ = class_name
    Cls.__module__ = "mjlab.fake"
    return Cls


# Fake mjlab observation functions (callables with __name__ and __module__)
def _make_mjlab_obs_func(name: str):
    def fn():
        pass

    fn.__name__ = name
    fn.__module__ = "mjlab.envs.mdp.observations"
    return fn


def _make_mjlab_term_func(name: str):
    def fn():
        pass

    fn.__name__ = name
    fn.__module__ = "mjlab.envs.mdp.terminations"
    return fn


# Fake mjlab config classes
FakeMjlabObsTermCfg = _make_mjlab_class(
    "ObservationTermCfg",
    func=None,
    params={},
    scale=None,
    clip=None,
    history_length=0,
    noise=None,
)

FakeMjlabObsGroupCfg = _make_mjlab_class(
    "ObservationGroupCfg",
    terms={},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=None,
)

FakeMjlabTermTermCfg = _make_mjlab_class(
    "TerminationTermCfg",
    func=None,
    params={},
    time_out=False,
)

FakeMjlabJointPositionActionCfg = _make_mjlab_class(
    "JointPositionActionCfg",
    entity_name="robot",
    clip=None,
    actuator_names=(".*",),
    scale=1.0,
    offset=0.0,
    use_default_offset=True,
    stiffness=None,
    damping=None,
)

FakeMjlabJointEffortActionCfg = _make_mjlab_class(
    "JointEffortActionCfg",
    entity_name="robot",
    clip=None,
    actuator_names=(".*",),
    scale=1.0,
    offset=0.0,
    stiffness=None,
    damping=None,
)

# `MyoMuscleActivationActionCfg` is standalone (no dataclass, no BaseActionCfg),
# so mirror only the fields the adapter inspects.
FakeMyoMuscleActivationActionCfg = _make_mjlab_class(
    "MyoMuscleActivationActionCfg",
    entity_name="robot",
    actuator_names=("m1", "m2"),
)

FakeMjlabSceneEntityCfg = _make_mjlab_class(
    "SceneEntityCfg",
    **{
        "name": "robot",
        "joint_names": None,
        "site_names": None,
    },
)

FakeMjlabMotionCommandCfg = _make_mjlab_class(
    "MotionCommandCfg",
    anchor_body_name="torso_link",
    body_names=("pelvis", "torso_link"),
    entity_name="robot",
)


# ===================================================================
# Tests: Observations
# ===================================================================


class TestAdaptObservations:
    def test_none_passthrough(self):
        assert adapt_observations(None) is None

    def test_mjswan_types_unchanged(self):
        obs_func = ObservationBinding("BaseLinearVelocity")
        group = ObservationGroupCfg(terms={"vel": ObservationTermCfg(func=obs_func)})
        result = adapt_observations({"policy": group})
        assert result is not None
        assert result["policy"] is group

    def test_mjlab_obs_term_converted(self):
        # `base_lin_vel` is a DSL term (ADR 0003) — adapter resolves to a callable.
        mjlab_func = _make_mjlab_obs_func("base_lin_vel")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func, params={"world_frame": True})
        mjlab_group = FakeMjlabObsGroupCfg(
            terms={"base_vel": mjlab_term},
            enable_corruption=True,
        )

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        group = result["policy"]
        assert isinstance(group, ObservationGroupCfg)
        assert "base_vel" in group.terms
        term = group.terms["base_vel"]
        assert isinstance(term, ObservationTermCfg)
        assert callable(term.func)
        assert term.params == {"world_frame": True}

    def test_mjlab_obs_scale_and_history(self):
        mjlab_func = _make_mjlab_obs_func("joint_pos_rel")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func, scale=0.5, history_length=3)
        mjlab_group = FakeMjlabObsGroupCfg(terms={"jp": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        term = result["policy"].terms["jp"]
        assert term.scale == 0.5
        assert term.history_length == 3

    def test_mjlab_asset_cfg_kept_intact_for_tracing(self):
        # A plain (non-Binding) func is traced to ONNX at build time (ADR 0005) via `func(env,
        # **params)` — params must reach the tracer unchanged, including the real `asset_cfg`
        # object mjlab's own function expects, not a flattened entity_name/joint_names stand-in.
        mjlab_func = _make_mjlab_obs_func("joint_pos_rel")
        asset_cfg = FakeMjlabSceneEntityCfg(
            name="robot", joint_names=("joint1", "joint2")
        )
        mjlab_term = FakeMjlabObsTermCfg(
            func=mjlab_func, params={"asset_cfg": asset_cfg}
        )
        mjlab_group = FakeMjlabObsGroupCfg(terms={"jp": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        term = result["policy"].terms["jp"]
        assert term.params["asset_cfg"] is asset_cfg

    def test_any_mjlab_obs_func_passes_through_for_tracing(self):
        # ADR 0005: there is no mjswan-side mirror to resolve by name — any mjlab function
        # (however unfamiliar) is passed straight through and traced to ONNX at build time.
        mjlab_func = _make_mjlab_obs_func("nonexistent_function")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func)
        mjlab_group = FakeMjlabObsGroupCfg(terms={"x": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        assert result["policy"].terms["x"].func is mjlab_func

    def test_multiple_groups(self):
        # The adapter resolves these to callables.
        f1 = _make_mjlab_obs_func("base_ang_vel")
        f2 = _make_mjlab_obs_func("projected_gravity")
        g1 = FakeMjlabObsGroupCfg(terms={"ang": FakeMjlabObsTermCfg(func=f1)})
        g2 = FakeMjlabObsGroupCfg(terms={"grav": FakeMjlabObsTermCfg(func=f2)})

        result = adapt_observations({"policy": g1, "critic": g2})
        assert result is not None
        assert callable(result["policy"].terms["ang"].func)
        assert callable(result["critic"].terms["grav"].func)

    def test_tracking_observation_functions_are_mapped(self):
        motion_anchor = _make_mjlab_obs_func("motion_anchor_pos_b")
        body_pos = _make_mjlab_obs_func("robot_body_pos_b")
        result = adapt_observations(
            {
                "policy": FakeMjlabObsGroupCfg(
                    terms={
                        "anchor": FakeMjlabObsTermCfg(func=motion_anchor),
                        "body": FakeMjlabObsTermCfg(func=body_pos),
                    }
                )
            }
        )
        # The adapter resolves these to callables, not ObservationBinding sentinels.
        assert result is not None
        assert callable(result["policy"].terms["anchor"].func)
        assert callable(result["policy"].terms["body"].func)


# ===================================================================
# Tests: Terminations
# ===================================================================


class TestAdaptTerminations:
    def test_none_passthrough(self):
        assert adapt_terminations(None) is None

    def test_mjswan_types_unchanged(self):
        term_func = TerminationBinding("TimeOut")
        cfg = TerminationTermCfg(func=term_func, time_out=True)
        result = adapt_terminations({"time_out": cfg})
        assert result is not None
        assert result["time_out"] is cfg

    def test_mjlab_term_converted(self):
        # The adapter resolves this to a callable, not a TerminationBinding sentinel.
        mjlab_func = _make_mjlab_term_func("bad_orientation")
        mjlab_cfg = FakeMjlabTermTermCfg(
            func=mjlab_func,
            params={"limit_angle": 1.0},
            time_out=False,
        )

        result = adapt_terminations({"fallen": mjlab_cfg})
        assert result is not None
        term = result["fallen"]
        assert isinstance(term, TerminationTermCfg)
        assert callable(term.func)
        assert term.params == {"limit_angle": 1.0}
        assert term.time_out is False

    def test_mjlab_time_out_flag(self):
        # `time_out` is a DSL term (ADR 0003) — resolved to a callable.
        mjlab_func = _make_mjlab_term_func("time_out")
        mjlab_cfg = FakeMjlabTermTermCfg(func=mjlab_func, time_out=True)

        result = adapt_terminations({"timeout": mjlab_cfg})
        assert result is not None
        assert result["timeout"].time_out is True
        assert callable(result["timeout"].func)

    def test_mjlab_term_keeps_asset_cfg_intact_for_tracing(self):
        # As with observations, a plain func's params reach the tracer unflattened.
        mjlab_func = _make_mjlab_term_func("bad_orientation")
        asset_cfg = FakeMjlabSceneEntityCfg(name="robot", body_names=("torso_link",))
        mjlab_cfg = FakeMjlabTermTermCfg(
            func=mjlab_func,
            params={"limit_angle": 1.0, "asset_cfg": asset_cfg},
            time_out=False,
        )

        result = adapt_terminations({"fallen": mjlab_cfg})
        assert result is not None
        term = result["fallen"]
        assert term.params == {"limit_angle": 1.0, "asset_cfg": asset_cfg}

    def test_any_mjlab_term_func_passes_through_for_tracing(self):
        # No mjswan-side mirror: any mjlab function passes straight through to the tracer.
        mjlab_func = _make_mjlab_term_func("nonexistent_term")
        mjlab_cfg = FakeMjlabTermTermCfg(func=mjlab_func)

        result = adapt_terminations({"x": mjlab_cfg})
        assert result is not None
        assert result["x"].func is mjlab_func


# ===================================================================
# Tests: Actions
# ===================================================================


class TestAdaptActions:
    def test_none_passthrough(self):
        assert adapt_actions(None) is None


class TestAdaptCommands:
    def test_none_passthrough(self):
        assert adapt_commands(None) is None

    def test_motion_command_cfg_converts_to_tracking_command(self):
        result = adapt_commands({"motion": FakeMjlabMotionCommandCfg()})
        assert result is not None
        command = result["motion"]
        assert command.term_name == "TrackingCommand"
        assert command.params["anchor_body_name"] == "torso_link"
        assert command.params["body_names"] == ["pelvis", "torso_link"]

    def test_mjswan_types_unchanged(self):
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        cfg = JointPositionActionCfg(scale=0.5)
        result = adapt_actions({"joint_pos": cfg})
        assert result is not None
        assert result["joint_pos"] is cfg

    def test_mjlab_joint_position_converted(self):
        mjlab_cfg = FakeMjlabJointPositionActionCfg(
            scale=0.25,
            offset=0.1,
            use_default_offset=True,
            stiffness=40.0,
            damping=2.5,
        )

        result = adapt_actions({"jp": mjlab_cfg})
        assert result is not None
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        action = result["jp"]
        assert isinstance(action, JointPositionActionCfg)
        assert action.scale == 0.25
        assert action.offset == 0.1
        assert action.use_default_offset is True
        assert action.stiffness == 40.0
        assert action.damping == 2.5

    def test_mjlab_joint_effort_converted(self):
        mjlab_cfg = FakeMjlabJointEffortActionCfg(scale=2.0)

        result = adapt_actions({"torque": mjlab_cfg})
        assert result is not None
        from mjswan.envs.mdp.actions import JointEffortActionCfg

        action = result["torque"]
        assert isinstance(action, JointEffortActionCfg)
        assert action.scale == 2.0

    def test_mjlab_unknown_action_warns(self):
        FakeUnknown = _make_mjlab_class(
            "SomeWeirdActionCfg",
            entity_name="robot",
            clip=None,
            actuator_names=(".*",),
            scale=1.0,
            offset=0.0,
        )
        mjlab_cfg = FakeUnknown()

        with pytest.warns(RuntimeWarning, match="no mjswan equivalent"):
            result = adapt_actions({"weird": mjlab_cfg})

        assert result is not None
        assert "weird" not in result

    def test_mixed_mjswan_and_mjlab(self):
        """Both mjswan-native and mjlab types in the same dict."""
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        mjswan_cfg = JointPositionActionCfg(scale=0.5)
        mjlab_cfg = FakeMjlabJointEffortActionCfg(scale=3.0)

        result = adapt_actions({"jp": mjswan_cfg, "torque": mjlab_cfg})
        assert result is not None
        assert result["jp"] is mjswan_cfg
        from mjswan.envs.mdp.actions import JointEffortActionCfg

        assert isinstance(result["torque"], JointEffortActionCfg)


# ===================================================================
# Tests: End-to-end serialization after adaptation
# ===================================================================


class TestAdaptedSerialization:
    """Ensure adapted plain-callable terms defer to ONNX tracing (ADR 0005)."""

    def test_adapted_obs_to_dict_requires_tracing(self):
        # A plain-callable func (mjlab's own, resolved by the adapter with no mirror lookup)
        # cannot be serialized via to_dict()/to_list() directly — it must be traced to ONNX
        # against a live env at build time (mjswan._onnx_build.serialize_observation_group).
        mjlab_func = _make_mjlab_obs_func("last_action")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func)
        mjlab_group = FakeMjlabObsGroupCfg(terms={"la": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        with pytest.raises(TypeError, match="serialize_observation_group"):
            result["policy"].to_list()

    def test_adapted_term_to_dict_requires_tracing(self):
        mjlab_func = _make_mjlab_term_func("root_height_below_minimum")
        mjlab_cfg = FakeMjlabTermTermCfg(
            func=mjlab_func,
            params={"minimum_height": 0.2},
        )

        result = adapt_terminations({"fallen": mjlab_cfg})
        assert result is not None
        with pytest.raises(TypeError, match="serialize_termination"):
            result["fallen"].to_dict()

    def test_adapted_action_serializes(self):
        mjlab_cfg = FakeMjlabJointPositionActionCfg(
            scale={"hip": 0.5, "knee": 0.3},
            stiffness=40.0,
            damping=2.5,
        )

        result = adapt_actions({"jp": mjlab_cfg})
        assert result is not None
        d = result["jp"].to_dict()
        assert d["type"] == "joint_position"
        assert d["scale"] == {"hip": 0.5, "knee": 0.3}
        assert d["stiffness"] == 40.0
        assert d["damping"] == 2.5


# ---------------------------------------------------------------------------
# Muscle action adaptation: MyoMuscleActivationActionCfg → MuscleActivationActionCfg
# ---------------------------------------------------------------------------


class TestMuscleActionAdaptation:
    """Adapt myosuite4's ``MyoMuscleActivationActionCfg`` to mjswan."""

    def test_adapted_to_muscle_activation_action_cfg(self):
        from mjswan.envs.mdp.actions import MuscleActivationActionCfg

        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1", "m2"),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        assert isinstance(result["muscles"], MuscleActivationActionCfg)

    def test_adapted_action_serializes_to_muscle_activation_type(self):
        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1", "m2"),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        d = result["muscles"].to_dict()
        assert d["type"] == "muscle_activation"

    def test_actuator_names_prefixed_with_entity_name(self):
        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1", "m2"),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        d = result["muscles"].to_dict()
        assert d["actuator_names"] == ["robot/m1", "robot/m2"]

    def test_normalize_defaults_to_true_when_source_lacks_field(self):
        # MyoMuscleActivationActionCfg has no `normalize` field and always applies the sigmoid
        # mapping in upstream; the adapted cfg must keep the mjswan default (normalize=True),
        # which serializes as the key being omitted (default-suppression).
        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1",),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        d = result["muscles"].to_dict()
        assert "normalize" not in d
        assert result["muscles"].normalize is True

    def test_default_scale_offset_preserved_when_source_omits_them(self):
        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1",),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        d = result["muscles"].to_dict()
        assert "scale" not in d
        assert "offset" not in d
        assert result["muscles"].scale == 1.0
        assert result["muscles"].offset == 0.0

    def test_class_name_alias_dispatch(self):
        # The adapter looks up the source class name in _ACTION_CLASS_ALIASES. If the source
        # class is renamed upstream, this test would catch the break by failing dispatch.
        mjlab_cfg = FakeMyoMuscleActivationActionCfg()
        assert type(mjlab_cfg).__name__ == "MyoMuscleActivationActionCfg"

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        assert "muscles" in result
