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
from mjlab.utils.lab_api.math import sample_uniform, wrap_to_pi

from mjswan import CommandBinding, register_command

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
