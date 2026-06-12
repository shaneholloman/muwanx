"""Tests for the declarative MDP DSL (ADR 0003)."""

from __future__ import annotations

import pytest

from mjswan.dsl import any_, param, trace_event, trace_observation, trace_termination
from mjswan.envs.mdp import events as event_fns
from mjswan.envs.mdp import observations as obs_fns
from mjswan.envs.mdp import terminations as term_fns
from mjswan.managers.event_manager import EventTermCfg
from mjswan.managers.observation_manager import ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg


class TestSymbolicEnv:
    """Symbolic env attribute access builds primitive-op nodes."""

    def test_entity_data_slot_known(self):
        from mjswan.dsl import SymbolicEnv

        env = SymbolicEnv()
        ref = env.entity("robot").data.root_ang_vel_b
        assert ref.node.op == "RootAngVelB"
        assert ref.node.attrs == {"entity": "robot"}

    def test_entity_data_slot_unknown_raises(self):
        from mjswan.dsl import SymbolicEnv

        env = SymbolicEnv()
        with pytest.raises(AttributeError, match="Unknown entity data slot"):
            _ = env.entity("robot").data.nonexistent_field

    def test_command_field_access(self):
        from mjswan.dsl import SymbolicEnv

        env = SymbolicEnv()
        ref = env.command("motion").anchor_pos
        assert ref.node.op == "CommandField"
        assert ref.node.attrs == {"command": "motion", "field": "anchor_pos"}


class TestArithmeticTracing:
    """Python operators build primitive-op nodes lazily."""

    def test_subtract_two_refs(self):
        from mjswan.dsl import SymbolicEnv

        env = SymbolicEnv()
        a = env.entity("robot").data.root_link_pos_w
        b = env.entity("robot").data.root_link_lin_vel_b
        ref = a - b
        assert ref.node.op == "Sub"
        assert len(ref.node.inputs) == 2

    def test_subtract_scalar_creates_const(self):
        from mjswan.dsl import SymbolicEnv

        env = SymbolicEnv()
        ref = env.entity("robot").data.root_ang_vel_b - 0.5
        assert ref.node.op == "Sub"
        assert ref.node.inputs[1].op == "Const"
        assert ref.node.inputs[1].attrs == {"value": 0.5}

    def test_abs_via_builtin(self):
        from mjswan.dsl import SymbolicEnv

        env = SymbolicEnv()
        ref = abs(env.entity("robot").data.root_ang_vel_b)
        assert ref.node.op == "Abs"

    def test_index_requires_static_int(self):
        from mjswan.dsl import SymbolicEnv

        env = SymbolicEnv()
        with pytest.raises(TypeError, match="static integer"):
            _ = env.entity("robot").data.root_ang_vel_b[1.5]  # type: ignore[index]


class TestTraceTermination:
    """End-to-end: traced function → JSON envelope."""

    def test_base_ang_vel_exceed_shape(self):
        def base_ang_vel_exceed(env, threshold):
            return any_(abs(env.entity("robot").data.root_ang_vel_b) > threshold)

        graph = trace_termination(base_ang_vel_exceed, {"threshold": 2.0})
        assert graph["kind"] == "termination"
        assert isinstance(graph["nodes"], list)
        assert all("op" in node and "out" in node for node in graph["nodes"])
        ops = [node["op"] for node in graph["nodes"]]
        assert "RootAngVelB" in ops
        assert "Abs" in ops
        assert "Gt" in ops
        assert "Any" in ops
        assert graph["output"] == graph["nodes"][-1]["out"]

    def test_explicit_param_creates_param_node(self):
        def term(env, threshold):
            del threshold
            return any_(env.entity("robot").data.root_ang_vel_b > param("threshold"))

        graph = trace_termination(term, {"threshold": 2.0})
        ops = [node["op"] for node in graph["nodes"]]
        assert "Param" in ops

    def test_non_noderef_return_raises(self):
        def bad_term(env):
            return env.entity("robot").data.root_ang_vel_b

        # Returning the NodeRef directly is fine; but returning a plain
        # Python value should raise.
        def really_bad(env):
            del env
            return 42

        with pytest.raises(TypeError, match="NodeRef"):
            trace_termination(really_bad, {})


class TestTermCfgDualPath:
    """ObservationTermCfg/TerminationTermCfg honour both legacy and DSL `func`."""

    def test_dsl_callable_emits_kind_termination_envelope(self):
        def term(env, threshold):
            return any_(abs(env.entity("robot").data.root_ang_vel_b) > threshold)

        cfg = TerminationTermCfg(func=term, params={"threshold": 1.5})
        out = cfg.to_dict()
        assert out["kind"] == "termination"
        assert "nodes" in out
        assert "name" not in out

    def test_legacy_termfunc_emits_named_entry(self):
        # `illegal_contact` is still a legacy TermFunc sentinel (unsupported,
        # but exercises the legacy serialization path with a fake func).
        from mjswan.envs.mdp.terminations import TermFunc

        cfg = TerminationTermCfg(
            func=TermFunc(ts_name="ExampleLegacy"), params={"limit_angle": 0.7}
        )
        out = cfg.to_dict()
        assert out["name"] == "ExampleLegacy"
        assert "kind" not in out

    def test_migrated_base_ang_vel_exceed_uses_dsl_path(self):
        """The migrated `term_fns.base_ang_vel_exceed` is now a callable
        and serializes via the DSL path."""
        cfg = TerminationTermCfg(
            func=term_fns.base_ang_vel_exceed, params={"threshold": 2.0}
        )
        out = cfg.to_dict()
        assert out["kind"] == "termination"


class TestMigratedTerminations:
    """The remaining terminations migrated off TermFunc sentinels trace to
    well-formed graphs referencing the expected primitive ops."""

    def _ops(self, func, params):
        graph = trace_termination(func, params)
        return [node["op"] for node in graph["nodes"]], graph

    def test_terrain_edge_reached_uses_spawn_and_step_count(self):
        ops, graph = self._ops(
            term_fns.terrain_edge_reached, {"half_x": 2.0, "half_y": 2.0}
        )
        assert graph["kind"] == "termination"
        assert "SpawnCapture" in ops
        assert "StepCount" in ops
        assert "RootLinkPosW" in ops

    def test_bad_anchor_pos_z_only_uses_tracking_sources(self):
        ops, _ = self._ops(term_fns.bad_anchor_pos_z_only, {"threshold": 0.5})
        assert "TrackingAnchorPos" in ops
        assert "TrackingCurrentAnchorPos" in ops
        assert "Gt" in ops

    def test_bad_anchor_ori_uses_quat_apply_inv(self):
        ops, _ = self._ops(term_fns.bad_anchor_ori, {"threshold": 0.5})
        assert "TrackingAnchorQuat" in ops
        assert "TrackingCurrentAnchorQuat" in ops
        assert "QuatApplyInv" in ops
        assert "ConstVec" in ops

    def test_bad_motion_body_pos_z_only_default_and_restricted(self):
        ops_default, _ = self._ops(
            term_fns.bad_motion_body_pos_z_only, {"threshold": 0.3}
        )
        assert "TrackingBodyPosZDeviationMax" in ops_default
        # Restricted body list flows into attrs.
        graph = trace_termination(
            term_fns.bad_motion_body_pos_z_only,
            {"threshold": 0.3, "body_names": ["pelvis", "torso"]},
        )
        dev_node = next(
            n for n in graph["nodes"] if n["op"] == "TrackingBodyPosZDeviationMax"
        )
        assert dev_node["attrs"]["body_names"] == ["pelvis", "torso"]

    def test_all_four_serialize_via_dsl_path_in_cfg(self):
        for func, params in [
            (term_fns.terrain_edge_reached, {"half_x": 1.0, "half_y": 1.0}),
            (term_fns.bad_anchor_pos_z_only, {"threshold": 0.5}),
            (term_fns.bad_anchor_ori, {"threshold": 0.5}),
            (term_fns.bad_motion_body_pos_z_only, {"threshold": 0.3}),
        ]:
            out = TerminationTermCfg(func=func, params=params).to_dict()
            assert out["kind"] == "termination"
            assert "name" not in out


class TestTraceObservation:
    def test_observation_envelope(self):
        from mjswan.dsl import SymbolicEnv  # noqa: F401 (import smoke)

        def obs(env):
            return env.entity("robot").data.root_link_lin_vel_b

        graph = trace_observation(obs, {})
        assert graph["kind"] == "observation"
        assert graph["nodes"][0]["op"] == "RootLinkLinVelB"

    def test_scale_clip_history_baked_into_graph(self):
        def obs(env):
            return env.entity("robot").data.root_link_lin_vel_b

        graph = trace_observation(obs, {}, scale=0.5, clip=(-2.0, 2.0), history_steps=3)
        ops = [n["op"] for n in graph["nodes"]]
        # mjlab order: compute -> clip -> scale -> history.
        assert ops.index("Clip") < ops.index("Mul") < ops.index("History")
        hist = next(n for n in graph["nodes"] if n["op"] == "History")
        assert hist["attrs"]["steps"] == 3
        assert graph["output"] == graph["nodes"][-1]["out"]

    def test_vector_scale_uses_constvec(self):
        def obs(env):
            return env.entity("robot").data.root_link_lin_vel_b

        graph = trace_observation(obs, {}, scale=[0.1, 0.2, 0.3])
        cv = next(n for n in graph["nodes"] if n["op"] == "ConstVec")
        assert cv["attrs"]["values"] == [0.1, 0.2, 0.3]

    def test_transpose_sets_history_interleaved(self):
        # last_action with history + transpose → interleaved (joint-major) History.
        cfg = ObservationTermCfg(
            func=obs_fns.last_action, history_length=3, params={"transpose": True}
        )
        out = cfg.to_dict()
        hist = next(n for n in out["nodes"] if n["op"] == "History")
        assert hist["attrs"]["steps"] == 3
        assert hist["attrs"]["interleaved"] is True

        # Without transpose → step-major (no interleaved attr).
        out2 = ObservationTermCfg(func=obs_fns.last_action, history_length=3).to_dict()
        hist2 = next(n for n in out2["nodes"] if n["op"] == "History")
        assert "interleaved" not in hist2["attrs"]

    def test_previous_actions_is_last_action_alias(self):
        assert obs_fns.previous_actions is obs_fns.last_action


class TestMigratedObservations:
    """Migrated observation built-ins trace to graphs with the expected ops."""

    def _ops(self, func, params):
        return [n["op"] for n in trace_observation(func, params)["nodes"]]

    def test_joint_pos_rel_all_subtracts_default(self):
        ops = self._ops(obs_fns.joint_pos_rel, {})
        assert "JointPos" in ops and "DefaultJointPos" in ops and "Sub" in ops

    def test_joint_pos_rel_with_explicit_default_uses_constvec(self):
        ops = self._ops(
            obs_fns.joint_pos_rel,
            {"joint_names": ["a", "b"], "default_joint_pos": [0.1, 0.2]},
        )
        assert "JointPos" in ops and "ConstVec" in ops and "DefaultJointPos" not in ops

    def test_joint_vel_rel(self):
        assert "JointVel" in self._ops(obs_fns.joint_vel_rel, {})

    def test_last_action(self):
        assert "PrevAction" in self._ops(obs_fns.last_action, {})

    def test_generated_commands(self):
        ops = self._ops(obs_fns.generated_commands, {"command_name": "velocity"})
        assert "CommandValue" in ops

    def test_builtin_sensor(self):
        assert "Sensor" in self._ops(obs_fns.builtin_sensor, {"sensor_name": "imu"})

    def test_joint_pos_cos_sin_pattern(self):
        # joint_pos_cos_sin is a task-side term now (Cartpole); verify the
        # cos/sin/concat composition still traces from core primitives.
        def joint_pos_cos_sin(env, *, joint_name, entity_name="robot", **_):
            from mjswan.dsl import concat, cos, joint_pos, sin

            angle = joint_pos([joint_name], entity=entity_name)
            return concat([cos(angle), sin(angle)])

        ops = self._ops(joint_pos_cos_sin, {"joint_name": "hinge"})
        assert "Cos" in ops and "Sin" in ops and "Concat" in ops

    def test_motion_anchor_pos_b(self):
        ops = self._ops(obs_fns.motion_anchor_pos_b, {})
        assert "TrackingAnchorPos" in ops and "QuatApplyInv" in ops

    def test_motion_anchor_ori_b(self):
        ops = self._ops(obs_fns.motion_anchor_ori_b, {})
        assert "QuatToRot6d" in ops and "QuatMul" in ops

    def test_robot_body_pos_b_unrolls_over_bodies(self):
        graph = trace_observation(
            obs_fns.robot_body_pos_b, {"body_names": ["pelvis", "torso", "head"]}
        )
        ops = [n["op"] for n in graph["nodes"]]
        # One BodyPos per body, all concatenated.
        assert ops.count("BodyPos") == 3
        assert "Concat" in ops

    def test_robot_body_ori_b_unrolls_over_bodies(self):
        graph = trace_observation(
            obs_fns.robot_body_ori_b, {"body_names": ["pelvis", "torso"]}
        )
        ops = [n["op"] for n in graph["nodes"]]
        assert ops.count("BodyQuat") == 2
        assert ops.count("QuatToRot6d") == 2

    def test_task_specific_dsl_obs_registers_and_traces(self):
        # Task-specific terms (e.g. ee_to_object_distance) live in the task and
        # register a DSL builder callable — they are NOT core built-ins.
        # Verify such a callable traces to a declarative graph (ADR 0003).
        def ee_to_object_distance(env, *, object_name, site_name=None, **_):
            from mjswan.dsl import body_pos, quat_apply_inv, site_pos

            base_quat = env.entity("robot").data.root_link_quat_w
            return quat_apply_inv(
                base_quat, body_pos(object_name) - site_pos(site_name or "")
            )

        ops = self._ops(
            ee_to_object_distance, {"object_name": "cube", "site_name": "robot/ee"}
        )
        assert "SitePos" in ops and "BodyPos" in ops
        assert "RootLinkQuatW" in ops and "QuatApplyInv" in ops

    def test_registered_dsl_callable_resolves_via_adapter(self):
        # register_obs_func with a DSL callable → adapter resolves it (no core
        # built-in needed) → ObservationTermCfg serializes a graph envelope.
        from mjswan.adapters.mjlab_adapter import _adapt_obs_func
        from mjswan.envs.mdp import observations as obs_mod

        def my_task_obs(env, **_):
            return env.entity("robot").data.root_link_lin_vel_b

        try:
            obs_mod._custom_registry["my_task_obs"] = my_task_obs

            class _FakeMjlabFunc:
                __name__ = "my_task_obs"

            resolved = _adapt_obs_func(_FakeMjlabFunc())
            assert resolved is my_task_obs
            out = ObservationTermCfg(func=resolved).to_dict()
            assert out["kind"] == "observation"
            assert "name" not in out
        finally:
            obs_mod._custom_registry.pop("my_task_obs", None)


class TestMigratedEvents:
    """Event builders trace to mutation-descriptor envelopes (ADR 0003)."""

    def test_randomize_terrain_is_noop(self):
        env = trace_event(event_fns.randomize_terrain, {})
        assert env == {"kind": "event", "mutations": []}

    def test_reset_joints_by_offset_emits_qpos_and_qvel(self):
        env = trace_event(
            event_fns.reset_joints_by_offset,
            {"position_range": [-0.1, 0.1], "velocity_range": [-0.2, 0.2]},
        )
        assert env["kind"] == "event"
        targets = [m["target"] for m in env["mutations"]]
        assert targets == ["joint_qpos", "joint_qvel"]
        qpos = env["mutations"][0]
        assert qpos["op"] == "add"
        assert qpos["sample"] == {"dist": "uniform", "low": -0.1, "high": 0.1}
        assert qpos["clip_to_limits"] is True

    def test_reset_joints_by_offset_carries_selection(self):
        env = trace_event(
            event_fns.reset_joints_by_offset,
            {
                "position_range": [0.0, 0.0],
                "velocity_range": [0.0, 0.0],
                "entity_name": "robot",
                "joint_names": ["hip", "knee"],
            },
        )
        sel = env["mutations"][0]["select"]
        assert sel["entity_name"] == "robot"
        assert sel["joint_names"] == ["hip", "knee"]

    def test_reset_root_state_uniform_pos_and_yaw(self):
        env = trace_event(
            event_fns.reset_root_state_uniform,
            {"pose_range": {"x": [-1.0, 1.0], "yaw": [-3.14, 3.14]}},
        )
        targets = [m["target"] for m in env["mutations"]]
        assert "freejoint_pos" in targets
        assert "freejoint_yaw" in targets
        yaw = next(m for m in env["mutations"] if m["target"] == "freejoint_yaw")
        assert yaw["op"] == "compose"

    def test_reset_root_state_uniform_no_yaw_when_absent(self):
        env = trace_event(
            event_fns.reset_root_state_uniform, {"pose_range": {"x": [-1.0, 1.0]}}
        )
        targets = [m["target"] for m in env["mutations"]]
        assert targets == ["freejoint_pos"]

    def test_event_cfg_dual_path(self):
        # DSL builder → event envelope; EventFunc binding (e.g. a task ts_src
        # event) → named entry.
        dsl = EventTermCfg(func=event_fns.reset_joints_by_offset).to_dict()
        assert dsl["kind"] == "event"
        assert "name" not in dsl

        legacy = EventTermCfg(
            func=event_fns.EventFunc(ts_name="TaskTsSrcEvent")
        ).to_dict()
        assert legacy["name"] == "TaskTsSrcEvent"
        assert "kind" not in legacy
