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
from mjswan.wandb_io import fetch_motion_npz_from_wandb_run

# The terminations need no registration: they trace straight from mjlab's own functions.


def setup_builder() -> mjswan.Builder:
    """Create the builder for the G1 spinkick tracking demo."""
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)

    run_path = "ttktjmt-org/mjlab/mayq0rtd"
    task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

    # mjlab's tracking config ships `motion_file=""` for the caller to fill, so the clip has
    # to land on disk before `add_scene_mjlab` constructs the tracing env.
    env_cfg = load_env_cfg(task_id, play=True)
    motion_name, motion_bytes = fetch_motion_npz_from_wandb_run(run_path)
    motion_path = example_dir / "artifacts" / f"{motion_name}.npz"
    motion_path.parent.mkdir(exist_ok=True)
    motion_path.write_bytes(motion_bytes)
    env_cfg.commands["motion"].motion_file = str(motion_path)

    builder = mjswan.Builder()

    project = builder.add_project(name="mjlab Spinkick")
    # `add_policy_wandb` reuses this clip; it re-downloads only if the path is not a file.
    scene = project.add_scene_mjlab(task_id, play=True, env_cfg=env_cfg)

    # Observations, commands, actions and terminations all come from the scene's own
    # `env_cfg` — the one edited above — and `task_id` from the scene. Nothing to restate.
    scene.add_policy_wandb(run_path)

    return builder


def main() -> None:
    """Build and optionally launch the G1 spinkick demo."""
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
