"""MuscleMimic Fullbody motion-tracking demo.

This example exercises mjswan's mjlab tracking playback path end-to-end for the
myosuite ``myoMimicFullbody-v0`` task:
- registers the task via myosuite's mjlab bootstrap (needs a motion clip on disk)
- pulls a sample clip from the gated ``amathislab/musclemimic-retargeted`` HF dataset
- loads policy checkpoints exported from a W&B training run
- mjswan replaces the registration clip with the run's motion artifact at export time

See ``README.md`` for prerequisites (HF auth, musclemimic_models, W&B login).
"""

from __future__ import annotations

import os
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    __package__ = "examples.mjlab.musclemimic"

import mjlab.tasks  # noqa: F401 - populates the mjlab task registry
from huggingface_hub import hf_hub_download
from mjlab.tasks.registry import load_env_cfg
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    bootstrap_myosuite_mjlab_registry,
)

import mjswan

CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"


def _bootstrap_mimic_task() -> None:
    """Download a registration clip and register ``myoMimicFullbody-v0``.

    The clip below is only used so the task appears in mjlab's registry; mjswan
    replaces ``env_cfg.commands["motion"].motion_file`` with the W&B run's
    motion artifact before exporting.
    """
    clip_path = Path(
        hf_hub_download(
            repo_id=CLIP_REPO_ID,
            filename=CLIP_FILENAME,
            repo_type="dataset",
        )
    )
    os.environ["MIMIC_CLIP"] = str(clip_path)
    bootstrap_myosuite_mjlab_registry()


def setup_builder() -> mjswan.Builder:
    """Create the builder for the MuscleMimic Fullbody tracking demo."""
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)

    _bootstrap_mimic_task()

    run_path = "ttktjmt-org/mjlab/zyklrroq"
    task_id = "myoMimicFullbody-v0"

    builder = mjswan.Builder(debug=True)

    project = builder.add_project(name="MuscleMimic Fullbody")
    scene = project.add_mjlab_scene(task_id, play=True)

    env_cfg = load_env_cfg(task_id, play=True)
    # Minimal config first: skip observations/terminations to confirm the rest of
    # the pipeline (scene, W&B fetch, motion artifact, ONNX export) succeeds.
    # Custom TS observations come next.
    scene.add_policy_from_wandb(
        run_path,
        task_id=task_id,
        commands=env_cfg.commands,
        actions=env_cfg.actions,
    )

    return builder


def main() -> None:
    """Build and optionally launch the MuscleMimic Fullbody demo."""
    app = setup_builder().build()
    app.launch()


if __name__ == "__main__":
    main()
