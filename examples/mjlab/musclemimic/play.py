"""Wrapper around ``mjlab play`` for myoMimicFullbody-v0.

``mjlab play`` validates the task ID against the registry at argument-parse
time, before any bootstrap code can run.  myoMimicFullbody-v0 is only
registered after ``bootstrap_myosuite_mjlab_registry()`` is called (which
itself requires a motion clip on disk), so the task never appears in the
standard choices list.

This script bootstraps the task first, then delegates to
``mjlab.scripts.play.run_play`` directly, bypassing the registry check.

Usage (mirrors ``uv run play myoMimicFullbody-v0 ...``):
    uv run python examples/mjlab/musclemimic/play.py [OPTIONS]
    uv run python examples/mjlab/musclemimic/play.py --help

All PlayConfig options (--wandb-run-path, --agent, --viewer, etc.) are
forwarded as-is.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    __package__ = "examples.mjlab.musclemimic"

import mjlab.tasks  # noqa: F401 - populates the base mjlab task registry
import tyro
from huggingface_hub import hf_hub_download
from mjlab.scripts.play import PlayConfig, run_play
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    bootstrap_myosuite_mjlab_registry,
)

TASK_ID = "myoMimicFullbody-v0"
# WANDB_RUN_PATH = "ttktjmt-org/mjlab/v3q5ete6"
# WANDB_RUN_PATH = "ttktjmt-org/mjlab/srir3zxo"
WANDB_RUN_PATH = "ttktjmt-org/mjlab/y38ix8cv"
CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"


def _bootstrap() -> None:
    clip_path = Path(
        hf_hub_download(
            repo_id=CLIP_REPO_ID,
            filename=CLIP_FILENAME,
            repo_type="dataset",
        )
    )
    os.environ["MIMIC_CLIP"] = str(clip_path)
    bootstrap_myosuite_mjlab_registry()


def main() -> None:
    _bootstrap()

    cfg = tyro.cli(
        PlayConfig,
        default=PlayConfig(wandb_run_path=WANDB_RUN_PATH),
        prog=f"{Path(sys.argv[0]).name}",
    )
    run_play(TASK_ID, cfg)


if __name__ == "__main__":
    main()
