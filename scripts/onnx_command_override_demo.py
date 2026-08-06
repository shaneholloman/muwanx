#!/usr/bin/env python
"""Demonstrate the examples-side trace-friendly override for a blocked command.

Companion brief §3a follow-up. `UniformVelocityCommand` does not trace
(`aten::uniform` from `r.uniform_()`, and `.nonzero()`+branch control flow). The
fix that needs no upstream mjlab change and no native-TS engine work: the task
author supplies a **trace-friendly, numerically-equivalent override** of the
term's `_resample_command`/`_update_command` — using `sample_uniform` (spyable)
and `torch.where` (branch-free) instead — and swaps it in before compiling, per
ADR 0003's "authors write mjlab-style Python terms".

This demo binds such an override onto a live `UniformVelocityCommand` and shows it
now traces to ONNX and holds graph-vs-override parity. It covers the core resample
(lin/ang velocity, standing mask) and the standing-zero update; heading tracking
reads runtime robot state (`heading_w`) and needs dynamic-slot Command support
(next increment), so it is intentionally omitted here.

Usage (headless)::

    MUJOCO_GL=disable uv run python scripts/onnx_command_override_demo.py
"""

from __future__ import annotations

import os
import sys
import types

import torch

os.environ.setdefault("MUJOCO_GL", "disable")

# Into this module's globals, so the RNG spy (which patches `__globals__`) sees the draws.
from mjlab.utils.lab_api.math import sample_uniform, wrap_to_pi  # noqa: E402


def tf_resample_command(self, env_ids) -> None:
    """Trace-friendly `_resample_command`: sample_uniform + full-tensor build.

    Numerically equivalent to UniformVelocityCommand's core resample at N=1
    (resamples all envs; the resample_mask gate in the traced module selects
    which envs actually take the new value). Covers lin/ang velocity, heading
    target + is_heading mask, and standing mask.
    """
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


def tf_update_command(self) -> None:
    """Trace-friendly `_update_command`: heading tracking + standing, masked.

    Heading tracking reads runtime robot state (`heading_w`) — the dynamic-slot
    read this demo exists to exercise — and applies it via `torch.where` instead
    of mjlab's `.nonzero()`+index-assign.
    """
    # Heading tracking: steer ang_vel_z toward the heading target, is_heading envs.
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
    # Standing envs: zero the command.
    standing = self.is_standing_env.reshape(-1, 1)
    self.vel_command_b = torch.where(standing, torch.zeros_like(vc), vc)


def main() -> int:
    import json

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg

    from mjswan.compile import (
        command_config,
        run_command_parity,
        trace_command_term,
        validate_command_config,
    )

    env = ManagerBasedRlEnv(unitree_go1_flat_env_cfg(play=True), device="cpu")
    env.reset()
    term = env.command_manager._terms["twist"]

    # Swap in the trace-friendly override (what an examples-side author would do).
    term._resample_command = types.MethodType(tf_resample_command, term)
    term._update_command = types.MethodType(tf_update_command, term)
    print(f"overrode {type(term).__name__}._resample_command / _update_command")

    state_fields = [
        "vel_command_b",
        "heading_target",
        "is_heading_env",
        "is_standing_env",
    ]
    report = run_command_parity(
        term,
        state_fields=state_fields,
        name="twist",
        command_field="vel_command_b",
        n_draws=16,
    )
    status = "OK  " if report.passed else "FAIL"
    print(
        f"[{status}] twist (override)  command  rand_dim={report.rand_dim} "
        f"max|Δ|={report.max_abs_diff:.2e} over {report.steps_checked} draws  "
        f"({report.note})"
    )

    # End-to-end: emit the OnnxCommand policy.json config from the real trace.
    export = trace_command_term(
        term, state_fields, name="twist", command_field="vel_command_b"
    )
    # A velocity task author supplies the UI descriptor (mirrors mjlab create_gui).
    ui = {
        "controls": [
            {"type": "checkbox", "name": "enabled", "label": "Joystick"},
            {
                "type": "slider",
                "name": "lin_vel_x",
                "label": "X",
                "enabled_when": "enabled",
            },
            {
                "type": "slider",
                "name": "lin_vel_y",
                "label": "Y",
                "enabled_when": "enabled",
            },
            {
                "type": "slider",
                "name": "ang_vel_z",
                "label": "Yaw",
                "enabled_when": "enabled",
            },
            {"type": "button", "name": "zero", "label": "Zero"},
        ]
    }
    cfg = command_config(
        export,
        onnx_ref="command/twist.onnx",
        resampling_time_range=tuple(term.cfg.resampling_time_range),
        debug_vis=bool(getattr(term.cfg, "debug_vis", False)),
        ui=ui,
    )
    errors = validate_command_config(cfg)
    print(f"policy.json OnnxCommand config (valid={not errors}):")
    print(json.dumps({k: v for k, v in cfg.items() if k != "ui"}, indent=2))
    if errors:
        print("CONFIG ERRORS:", errors)
        report.passed = False

    print("PASS" if report.passed else "FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
