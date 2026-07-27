"""Mjlab-specific custom command registrations for mjswan examples.

Import this module before calling ``builder.build()`` to register the
command term bindings used in mjlab tasks.

Both commands below are ONNX-traced (ADR 0005 §3): the mjlab cfg is built
(``cfg.build(env)``) and its ``_resample_command``/``_update_command`` traced
directly against the scene's live env, run in the browser by the shared
``OnnxCommand`` handler — there is no per-command TS class anymore.
``LiftingCommand``'s target-position marker (previously rendered by the
retired ``LiftingCommand.ts``) is now the generic ``OnnxCommand.viz``
mechanism instead of a hand-written class.
"""

from __future__ import annotations

import types
from typing import Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    wrap_to_pi,
)

from mjswan import CommandBinding, register_command
from mjswan.command import _serialize_motion_command

# ---------------------------------------------------------------------------
# LiftingCommand (Lift-Cube-Yam) — traces directly, no override needed.
# `_resample_command` writes the cube's pose/velocity via
# `write_root_link_pose_to_sim`/`write_root_link_velocity_to_sim`
# (captured as `entity_write`, brief §3b); `target_pos` is the only state.
# ---------------------------------------------------------------------------


def _lifting_viz(cfg: Any) -> dict[str, Any]:
    """Render `target_pos` as a sphere, colored from the task's own cfg.viz."""
    return {
        "field": "target_pos",
        "shape": "sphere",
        "radius": 0.03,
        "color": list(cfg.viz.target_color),
    }


register_command(
    "LiftingCommandCfg",
    CommandBinding(
        state_fields=["target_pos"],
        command_field="target_pos",
        viz=_lifting_viz,
    ),
)


# ---------------------------------------------------------------------------
# UniformVelocityCommand (velocity tasks) — needs a trace-friendly override:
# the real `_resample_command` uses tensor-method RNG (`r.uniform_()`, invisible
# to the sample_uniform spy) and `_update_command` uses `.nonzero()` + index
# assignment (data-dependent control flow). The override below is numerically
# equivalent at N=1, using `sample_uniform` (spyable) and `torch.where`
# (branch-free) instead (ADR 0005 §3a).
# ---------------------------------------------------------------------------


def _tf_resample_command(self: Any, env_ids: torch.Tensor) -> None:
    n = self.num_envs
    dev = self.device
    r = self.cfg.ranges
    vx = sample_uniform(r.lin_vel_x[0], r.lin_vel_x[1], (n, 1), device=dev)
    vy = sample_uniform(r.lin_vel_y[0], r.lin_vel_y[1], (n, 1), device=dev)
    wz = sample_uniform(r.ang_vel_z[0], r.ang_vel_z[1], (n, 1), device=dev)
    self.vel_command_b = torch.cat([vx, vy, wz], dim=-1)
    self.heading_target = sample_uniform(r.heading[0], r.heading[1], (n,), device=dev)
    is_heading = sample_uniform(0.0, 1.0, (n,), device=dev)
    self.is_heading_env = is_heading <= self.cfg.rel_heading_envs
    standing = sample_uniform(0.0, 1.0, (n,), device=dev)
    self.is_standing_env = standing <= self.cfg.rel_standing_envs


def _tf_update_command(self: Any) -> None:
    heading_err = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
    wz = torch.clip(
        self.cfg.heading_control_stiffness * heading_err,
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
    ).reshape(-1, 1)
    heading_mask = self.is_heading_env.reshape(-1, 1)
    vx = self.vel_command_b[:, 0:1]
    vy = self.vel_command_b[:, 1:2]
    wz_col = torch.where(heading_mask, wz, self.vel_command_b[:, 2:3])
    vc = torch.cat([vx, vy, wz_col], dim=-1)
    standing = self.is_standing_env.reshape(-1, 1)
    self.vel_command_b = torch.where(standing, torch.zeros_like(vc), vc)


def _bind_trace_friendly_velocity_override(term: Any) -> None:
    term._resample_command = types.MethodType(_tf_resample_command, term)
    term._update_command = types.MethodType(_tf_update_command, term)


def _velocity_ui(cfg: Any) -> dict[str, Any]:
    """Joystick UI descriptor with slider ranges from this task's own ``cfg.ranges``.

    A static dict can't be reused across tasks with different velocity limits
    (Go1 vs G1), so this is resolved per-task from the real mjlab cfg, same as
    the (retired) native ``_serialize_uniform_velocity_command`` did.
    """
    ranges = cfg.ranges
    return {
        "inputs": [
            {
                "type": "checkbox",
                "name": "enabled",
                "label": "Joystick",
                "default": False,
            },
            {
                "type": "slider",
                "name": "lin_vel_x",
                "label": "Forward Velocity",
                "min": ranges.lin_vel_x[0],
                "max": ranges.lin_vel_x[1],
                "step": 0.05,
                "default": max(ranges.lin_vel_x[0], min(0.5, ranges.lin_vel_x[1])),
                "enabled_when": "enabled",
            },
            {
                "type": "slider",
                "name": "lin_vel_y",
                "label": "Lateral Velocity",
                "min": ranges.lin_vel_y[0],
                "max": ranges.lin_vel_y[1],
                "step": 0.05,
                "default": max(ranges.lin_vel_y[0], min(0.0, ranges.lin_vel_y[1])),
                "enabled_when": "enabled",
            },
            {
                "type": "slider",
                "name": "ang_vel_z",
                "label": "Yaw Rate",
                "min": ranges.ang_vel_z[0],
                "max": ranges.ang_vel_z[1],
                "step": 0.05,
                "default": max(ranges.ang_vel_z[0], min(0.0, ranges.ang_vel_z[1])),
                "enabled_when": "enabled",
            },
            {"type": "button", "name": "zero", "label": "Zero"},
        ]
    }


register_command(
    "UniformVelocityCommandCfg",
    CommandBinding(
        state_fields=[
            "vel_command_b",
            "heading_target",
            "is_heading_env",
            "is_standing_env",
        ],
        command_field="vel_command_b",
        trace_override=_bind_trace_friendly_velocity_override,
        ui=_velocity_ui,
    ),
)


# ---------------------------------------------------------------------------
# MotionCommand (tracking tasks) — the motion player stays native (the clip
# lookup is a data lookup, not term math), but its *reference-state
# initialization* randomization is term math and is traced.
# ---------------------------------------------------------------------------

_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _range_tensor(
    ranges: dict[str, tuple[float, float]] | None, device: Any
) -> torch.Tensor:
    """mjlab's `range_list` -> (6, 2) tensor, missing axes meaning no offset."""
    ranges = ranges or {}
    return torch.tensor(
        [tuple(ranges.get(key, (0.0, 0.0))) for key in _POSE_KEYS],
        dtype=torch.float,
        device=device,
    )


def motion_rsi_offset(
    env: Any,
    env_ids: Any,
    *,
    asset_cfg: Any,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reference-state-initialization jitter from ``MotionCommand._resample_command``.

    mjlab perturbs the reference frame it is about to write; this reads the frame
    *already written* and perturbs it in place. Numerically the same — the offsets
    are added to the same values, and the clip still lands after the addition —
    but it needs no access to the motion clip, so the whole thing is ordinary term
    math over ``asset.data`` and traces to ONNX with the mechanisms Cartpole's
    resets and Go1's ``push_robot``/``reset_base`` already use.

    That is the point: the browser previously did this jitter in hand-written
    TypeScript off ``Math.random()``, which is neither mjlab's function nor
    replayable. Here it is mjlab's own ``sample_uniform``, so the draws become the
    graph's ``rand`` input, fed from the orchestrator's seeded PRNG (ADR 0005 §2),
    and the arithmetic is mjlab's rather than a paraphrase of it.

    The draws are ordered pose -> velocity -> joint to match mjlab's own order, so
    the two bodies read side by side.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.data.joint_pos.device

    # Root pose: xyz offset, plus a roll/pitch/yaw delta applied as a quaternion.
    pose_ranges = _range_tensor(pose_range, device)
    pose_samples = sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (1, 6), device=device
    )
    root_pos = asset.data.root_link_pos_w + pose_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    root_quat = quat_mul(orientations_delta, asset.data.root_link_quat_w)

    # Root velocity: linear and angular offsets, no rotation involved.
    velocity_ranges = _range_tensor(velocity_range, device)
    velocity_samples = sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (1, 6), device=device
    )
    root_lin_vel = asset.data.root_link_lin_vel_w + velocity_samples[:, 0:3]
    root_ang_vel = asset.data.root_link_ang_vel_w + velocity_samples[:, 3:6]

    # Joint positions: one offset per joint, then mjlab's clip to the soft limits.
    # The TypeScript this replaces omitted that clip, so a large enough jitter
    # could seed the episode outside the robot's own limits.
    joint_pos = asset.data.joint_pos + sample_uniform(
        joint_position_range[0],
        joint_position_range[1],
        asset.data.joint_pos.shape,
        device=device,
    )
    soft_limits = asset.data.soft_joint_pos_limits
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

    asset.write_joint_state_to_sim(joint_pos, asset.data.joint_vel, env_ids=env_ids)
    asset.write_root_link_pose_to_sim(
        torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
        torch.cat([root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
    )


def _motion_rsi_trace(cfg: Any) -> tuple[Any, dict[str, Any]] | None:
    """The reset graph for a `MotionCommandCfg`, or None if it jitters nothing.

    mjlab's own play-mode override clears `pose_range`/`velocity_range` and leaves
    `joint_position_range` at (-0.1, 0.1), so a deployed tracking policy normally
    gets exactly the joint jitter — but an author may keep any subset, and all
    three go through the same graph.
    """
    pose_range = dict(getattr(cfg, "pose_range", None) or {})
    velocity_range = dict(getattr(cfg, "velocity_range", None) or {})
    joint_position_range = tuple(getattr(cfg, "joint_position_range", (0.0, 0.0)))
    if not pose_range and not velocity_range and joint_position_range == (0.0, 0.0):
        return None
    return (
        motion_rsi_offset,
        {
            "asset_cfg": SceneEntityCfg(getattr(cfg, "entity_name", None) or "robot"),
            "pose_range": pose_range,
            "velocity_range": velocity_range,
            "joint_position_range": joint_position_range,
        },
    )


# `MotionCommandCfg` is already bound to the native `TrackingCommand` in
# `mjswan.command`; re-registering here adds the traced reset graph without
# disturbing that. The motion player stays native — the clip lookup is a data
# lookup — while the jitter around it becomes mjlab's own math in ONNX.
register_command(
    "MotionCommandCfg",
    CommandBinding(
        ts_name="TrackingCommand",
        serializer=_serialize_motion_command,
        reset_trace=_motion_rsi_trace,
    ),
)
