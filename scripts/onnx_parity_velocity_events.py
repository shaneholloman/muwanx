#!/usr/bin/env python
"""Trace Velocity-task Event bodies to ONNX and check parity (companion brief §a).

Exercises the dynamic-slot + ``entity_write`` event path on Go1-Velocity-Flat:
``push_robot`` (interval, reads live ``root_link_vel_w``, writes root velocity)
and the reset-mode events. Steps the env first so root velocity is non-zero,
then verifies each traceable event term against the live mjlab computation using
recorded-RNG replay.

Usage (headless)::

    MUJOCO_GL=disable uv run python scripts/onnx_parity_velocity_events.py

Exit code 0 = parity held for every traceable term; 1 = a discrepancy.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "disable")


def main() -> int:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.envs.mdp import push_by_setting_velocity
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg

    from mjswan.compile import run_parity

    # The light play env (N=1), plus `push_robot` which play mode pops — the clean exerciser of
    # the dynamic-slot + root-velocity entity_write path.
    cfg = unitree_go1_flat_env_cfg(play=True)
    cfg.events["push_robot"] = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )
    env = ManagerBasedRlEnv(cfg, device="cpu")
    report = run_parity(
        env,
        n_steps=20,  # step first so root_link_vel_w is non-zero for push_robot
        seed=0,
        event_modes=("interval", "reset"),
        n_event_draws=16,
        include_obs=False,  # obs need Command handling (brief §3), out of scope here
    )
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
