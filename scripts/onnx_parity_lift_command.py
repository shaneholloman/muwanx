#!/usr/bin/env python
"""Trace the LiftingCommand body to ONNX and check parity (companion brief §3b).

First stateful Command traced to ONNX: promotes ``target_pos`` to explicit graph
I/O, threads randomness through ``rand``, and captures the cube pose/velocity
``entity_write``. Validates against the live mjlab LiftingCommand on
``Mjlab-Lift-Cube-Yam`` (difficulty="dynamic").

Usage (headless)::

    MUJOCO_GL=disable uv run python scripts/onnx_parity_lift_command.py

Exit code 0 = parity held; 1 = a discrepancy.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "disable")


def main() -> int:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.manipulation.config.yam.env_cfgs import yam_lift_cube_env_cfg

    from mjswan.compile import run_command_parity

    cfg = yam_lift_cube_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()

    term = env.command_manager._terms["lift_height"]
    print(f"command term: {type(term).__name__}  difficulty={term.cfg.difficulty}")

    report = run_command_parity(
        term,
        state_fields=["target_pos"],
        name="lift_height",
        command_field="target_pos",
        n_draws=16,
    )
    status = "OK  " if report.passed else "FAIL"
    print(
        f"[{status}] lift_height  command  rand_dim={report.rand_dim} "
        f"max|Δ|={report.max_abs_diff:.2e} over {report.steps_checked} draws  "
        f"({report.note})"
    )
    print("PASS" if report.passed else "FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
