"""G1 spinkick motion-tracking demo.

This example exercises mjswan's mjlab tracking playback path end-to-end:
- MuJoCo scene from mjlab's play config
- policy checkpoints exported from a W&B training run
- reference motion auto-imported from the run's motion artifact
"""

from __future__ import annotations

import os
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    __package__ = "examples.mjlab.g1_spinkick"

import mjlab.tasks  # noqa: F401 - populates the mjlab task registry
from mjlab.tasks.registry import load_env_cfg

# `examples.mjlab.defaults.commands` registers the traced reset graph for
# `MotionCommandCfg` — the reference-state-initialization jitter (ADR 0005 §3).
# Without it the command still builds, bound to the native `TrackingCommand`, but
# with no jitter graph, so every episode would start from the unjittered reference
# frame where mjlab's play config asks for `joint_position_range=(-0.1, 0.1)`.
import examples.mjlab.defaults.commands  # noqa: F401
import mjswan

# This task's terminations (`anchor_pos`, `anchor_ori`, `ee_body_pos`, plus a
# native `time_out`) need no registration: under ADR 0005 they are traced straight
# from mjlab's own functions. They were declarative DSL built-ins before that.


def setup_builder() -> mjswan.Builder:
    """Create the builder for the G1 spinkick tracking demo."""
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)

    run_path = "ttktjmt-org/mjlab/mayq0rtd"
    task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

    builder = mjswan.Builder()

    project = builder.add_project(name="mjlab Spinkick")
    scene = project.add_scene_mjlab(task_id, play=True)

    env_cfg = load_env_cfg(task_id, play=True)
    scene.add_policy_wandb(
        run_path,
        task_id=task_id,
        observations={"policy": env_cfg.observations["actor"]},
        commands=env_cfg.commands,
        actions=env_cfg.actions,
        terminations=env_cfg.terminations,
    )

    return builder


def main() -> None:
    """Build and optionally launch the G1 spinkick demo."""
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
