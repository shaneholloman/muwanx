#!/usr/bin/env python
"""Empirically probe UniformVelocityCommand tracing (companion brief §3a, finding 13).

Attempts to trace the velocity `twist` command with `trace_command_term` and
reports the concrete failure mode, confirming why it is blocked (tensor-method
RNG + data-dependent control flow) rather than relying on source reading alone.

Usage (headless)::

    MUJOCO_GL=disable uv run python scripts/onnx_probe_velocity_command.py
"""

from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "disable")

_STATE_FIELDS = [
    "vel_command_b",
    "vel_command_w",
    "heading_target",
    "is_heading_env",
    "is_standing_env",
    "is_world_env",
    "is_forward_env",
]


def main() -> int:
    import torch

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg

    from mjswan.compile import run_command_parity
    from mjswan.compile.rng import DrawRecorder

    env = ManagerBasedRlEnv(unitree_go1_flat_env_cfg(play=True), device="cpu")
    env.reset()
    term = env.command_manager._terms["twist"]
    print(f"command term: {type(term).__name__}  heading={term.cfg.heading_command}")

    # (i) Does the RNG spy see the draws? UniformVelocityCommand uses r.uniform_().
    with DrawRecorder(term._resample_command) as rec:
        term._resample_command(torch.arange(term.num_envs))
    print(
        f"draws seen by sample_uniform spy: rand_dim={rec.rand_dim} "
        f"(0 confirms tensor-method RNG is invisible to the spy)"
    )

    # (ii) Does the body trace + hold parity? Expected to fail or mismatch.
    try:
        report = run_command_parity(
            term,
            state_fields=_STATE_FIELDS,
            name="twist",
            command_field="vel_command_b",
            n_draws=8,
        )
        verdict = "PASS" if report.passed else "FAIL (traced but parity broke)"
        print(f"trace_command_term: {verdict}  max|Δ|={report.max_abs_diff:.2e}")
    except Exception as exc:  # noqa: BLE001 — this probe records the failure mode
        print(
            f"trace_command_term raised: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0][:120]}"
        )
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
