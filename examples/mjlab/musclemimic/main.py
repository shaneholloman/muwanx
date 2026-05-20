"""MuscleMimic Fullbody motion-tracking demo.

This example exercises mjswan's muscle-driven mimic playback path for the
myosuite ``myoMimicFullbody-v0`` task:
- registers the task via myosuite's mjlab bootstrap (needs a motion clip on disk)
- pulls a sample clip from the gated ``amathislab/musclemimic-retargeted`` HF dataset
- loads policy checkpoints exported from a W&B training run
- bundles the clip into the browser app so TS observations can read it at runtime

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
from myosuite.integrations.musclemimic.fullbody_model import (
    FULLBODY_BODY2SITES_FOR_MIMIC,
)

import mjswan
from mjswan.envs.mdp.observations import ObsFunc, register_obs_func
from mjswan.motion import MotionConfig

CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"

# Control timestep for myoMimicFullbody-v0: sim_dt=0.002, decimation=5.
CTRL_DT = 0.01

# Tracked site names in model order (17 sites).
SITE_NAMES = list(FULLBODY_BODY2SITES_FOR_MIMIC.values())

# Path to the custom TS observation implementations bundled at build time.
_TS_SRC = str(Path(__file__).resolve().parent / "MimicObservations.ts")


def _register_mimic_observations() -> None:
    """Register mjswan TS sentinels for all myoMimicFullbody-v0 observation terms.

    The mimic task uses Python closures (all named ``_fn``) so the adapter
    cannot resolve them by function name.  We register by *term name* instead,
    which the adapter uses as a fallback when the function-name lookup fails.
    """
    register_obs_func("qpos", ObsFunc(ts_name="MimicQpos", ts_src=_TS_SRC))
    register_obs_func(
        "qvel",
        ObsFunc(ts_name="MimicQvel", ts_src=_TS_SRC, defaults={"ctrl_dt": CTRL_DT}),
    )
    register_obs_func("act", ObsFunc(ts_name="MimicAct", ts_src=_TS_SRC))
    register_obs_func(
        "mimic_site_pos",
        ObsFunc(
            ts_name="MimicSitePos",
            ts_src=_TS_SRC,
            defaults={"site_names": SITE_NAMES},
        ),
    )
    register_obs_func(
        "mimic_site_target",
        ObsFunc(
            ts_name="MimicSiteTarget",
            ts_src=_TS_SRC,
            defaults={"site_names": SITE_NAMES, "ctrl_dt": CTRL_DT},
        ),
    )
    register_obs_func(
        "mimic_site_err",
        ObsFunc(
            ts_name="MimicSiteErr",
            ts_src=_TS_SRC,
            defaults={"site_names": SITE_NAMES, "ctrl_dt": CTRL_DT},
        ),
    )
    register_obs_func(
        "clip_ref_qpos",
        ObsFunc(
            ts_name="MimicClipRefQpos",
            ts_src=_TS_SRC,
            defaults={"ctrl_dt": CTRL_DT},
        ),
    )
    register_obs_func(
        "clip_ref_qvel",
        ObsFunc(
            ts_name="MimicClipRefQvel", ts_src=_TS_SRC, defaults={"ctrl_dt": CTRL_DT}
        ),
    )
    register_obs_func(
        "clip_phase",
        ObsFunc(
            ts_name="MimicClipPhase",
            ts_src=_TS_SRC,
            defaults={"ctrl_dt": CTRL_DT},
        ),
    )
    register_obs_func(
        "mimic_lookahead",
        ObsFunc(
            ts_name="MimicLookahead",
            ts_src=_TS_SRC,
            defaults={"k": 5, "stride": 20, "ctrl_dt": CTRL_DT, "n_clip_sites": 17},
        ),
    )


def _bootstrap_mimic_task() -> Path:
    """Download a registration clip and register ``myoMimicFullbody-v0``.

    Returns the path to the downloaded clip (used to bundle into the app).
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
    return clip_path


def setup_builder() -> mjswan.Builder:
    """Create the builder for the MuscleMimic Fullbody tracking demo."""
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)

    _register_mimic_observations()
    clip_path = _bootstrap_mimic_task()

    run_path = "ttktjmt-org/mjlab/zyklrroq"
    task_id = "myoMimicFullbody-v0"

    builder = mjswan.Builder(debug=True)

    project = builder.add_project(name="MuscleMimic Fullbody")
    scene = project.add_mjlab_scene(task_id, play=True)

    env_cfg = load_env_cfg(task_id, play=True)

    # Pass observations= to include the mimic-specific TS observation classes.
    # mimic_lookahead is registered as unsupported and will be skipped by the builder.
    policy_handles = scene.add_policy_from_wandb(
        run_path,
        task_id=task_id,
        observations={"policy": env_cfg.observations["actor"]},
        commands=env_cfg.commands,
        actions=env_cfg.actions,
    )

    # Load clip frame 0 to initialize the simulation at a valid walking pose.
    # The policy was trained with RSI; starting from the keyframe (T-pose) is
    # too far outside the training distribution.
    import numpy as np

    clip_npz = np.load(clip_path, allow_pickle=True)
    initial_qpos = clip_npz["qpos"][0].tolist()
    initial_qvel = clip_npz["qvel"][0].tolist()

    # Bundle the motion clip NPZ into every policy so the TS observations can
    # fetch it at runtime via runner.getConfig().motions.
    mimic_clip = MotionConfig(name="mimic_clip", source=str(clip_path))
    for handle in policy_handles:
        handle._config.motions.append(mimic_clip)
        handle._config.initial_qpos = initial_qpos
        handle._config.initial_qvel = initial_qvel
        # Muscle policies have no joint transmission, so policy_joint_names is
        # empty. Set policy_num_actions explicitly from the ONNX output shape so
        # the TS runtime knows how many actuator outputs to expect.
        output_shape = handle._config.model.graph.output[0].type.tensor_type.shape
        num_actions = output_shape.dim[1].dim_value
        if num_actions > 0:
            handle._config.policy_num_actions = num_actions

    return builder


def main() -> None:
    """Build and optionally launch the MuscleMimic Fullbody demo."""
    app = setup_builder().build()
    app.launch()


if __name__ == "__main__":
    main()
