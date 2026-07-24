#!/usr/bin/env python
"""Run the ADR 0005 Phase-1 ONNX parity harness on mjlab's Cartpole task.

Exports every value-returning observation term of ``Mjlab-Cartpole-Balance`` to
ONNX and asserts that ``onnxruntime`` reproduces the live mjlab env output within
tolerance for every term, every step.

Usage (headless)::

    MUJOCO_GL=disable uv run python scripts/onnx_parity_cartpole.py [n_steps]

Exit code 0 = parity held; 1 = a discrepancy was found.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "disable")


def main() -> int:
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 64

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

    from mjswan.compile import run_parity

    cfg = cartpole_balance_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    report = run_parity(env, obs_group="actor", n_steps=n_steps, seed=0)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
