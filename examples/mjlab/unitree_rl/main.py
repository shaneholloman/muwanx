"""Unitree RL mjlab demo.

Follow the instructions in the README to run this demo.
"""

import pathlib
import sys

import mjlab.tasks  # noqa: F401 - populates the mjlab task registry
import mujoco
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg

import mjswan

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Tracking terminations (bad_anchor_pos_z_only, bad_anchor_ori,
# bad_motion_body_pos_z_only, base_ang_vel_exceed) are declarative built-ins
# in mjswan.envs.mdp.terminations — no registration needed.


def setup_builder() -> mjswan.Builder:
    """Create the builder for the unitree_rl_mjlab demo."""

    run_paths = ["ttktjmt-org/mjlab/l3tgm74z", "ttktjmt-org/mjlab/7m1ycqsn"]
    task_id = "Unitree-G1-Tracking-No-State-Estimation"

    builder = mjswan.Builder()

    project = builder.add_project(name="Unitree RL")
    scene = project.add_mjlab_scene(task_id, play=True)

    # Customize skybox
    mjspec = scene._config.spec
    assert mjspec is not None
    mjspec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.6, 0.8, 0.9],
        rgb2=[0.9, 0.9, 0.9],
        width=512,
        height=512,
    )

    env_cfg = load_env_cfg(task_id, play=True)
    scene.add_policy_from_wandb(
        run_paths,
        task_id=task_id,
        observations={"policy": env_cfg.observations["actor"]},
        commands=env_cfg.commands,
        actions=env_cfg.actions,
        terminations=env_cfg.terminations,
    )

    return builder


def main() -> None:
    """Build and launch the unitree_rl_mjlab demo."""
    app = setup_builder().build()
    app.launch()


if __name__ == "__main__":
    main()
