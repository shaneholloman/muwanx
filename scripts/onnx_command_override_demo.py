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

# Imported into THIS module's globals so the RNG spy (which patches the override
# function's __globals__) can see and replay these draws.
from mjlab.utils.lab_api.math import sample_uniform  # noqa: E402


def tf_resample_command(self, env_ids) -> None:
    """Trace-friendly `_resample_command`: sample_uniform + full-tensor build.

    Numerically equivalent to UniformVelocityCommand's core resample at N=1
    (resamples all envs; the resample_mask gate in the traced module selects
    which envs actually take the new value).
    """
    n = self.num_envs
    dev = self.device
    r = self.cfg.ranges
    vx = sample_uniform(r.lin_vel_x[0], r.lin_vel_x[1], (n, 1), device=dev)
    vy = sample_uniform(r.lin_vel_y[0], r.lin_vel_y[1], (n, 1), device=dev)
    wz = sample_uniform(r.ang_vel_z[0], r.ang_vel_z[1], (n, 1), device=dev)
    self.vel_command_b = torch.cat([vx, vy, wz], dim=-1)
    standing = sample_uniform(0.0, 1.0, (n,), device=dev)
    self.is_standing_env = standing <= self.cfg.rel_standing_envs


def tf_update_command(self) -> None:
    """Trace-friendly `_update_command`: zero standing envs, masked (no nonzero)."""
    mask = self.is_standing_env.reshape(-1, 1)
    self.vel_command_b = torch.where(
        mask, torch.zeros_like(self.vel_command_b), self.vel_command_b
    )


def main() -> int:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg

    from mjswan.compile import run_command_parity

    env = ManagerBasedRlEnv(unitree_go1_flat_env_cfg(play=True), device="cpu")
    env.reset()
    term = env.command_manager._terms["twist"]

    # Swap in the trace-friendly override (what an examples-side author would do).
    term._resample_command = types.MethodType(tf_resample_command, term)
    term._update_command = types.MethodType(tf_update_command, term)
    print(f"overrode {type(term).__name__}._resample_command / _update_command")

    report = run_command_parity(
        term,
        state_fields=["vel_command_b", "is_standing_env"],
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
    print("PASS" if report.passed else "FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
