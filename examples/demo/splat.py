"""Gaussian Splat Background Demo

Demonstrates how to use add_splat() to attach a real-world Gaussian Splat
background to a scene.

Run with:
    uv run splat
"""

import os
from pathlib import Path

import mujoco
import onnx

import mjswan
from mjswan.envs.mdp import observations as obs_fns
from mjswan.envs.mdp import terminations as term_fns
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg

SPLAT_URLs = [
    "https://cdn.marble.worldlabs.ai/be100eec-f02e-491d-899e-d702652d424d/cb27e09c-e2ca-46c7-8abf-bcd24d2bf9ed_ceramic_500k.spz",
    "https://cdn.marble.worldlabs.ai/09eaec3b-9114-455a-b7f1-da4d037cc511/660e6ce6-959c-42fb-8a9d-66178cb84f4d_ceramic.spz",
]

# fmt: off
_G1_SCALE = {
    "left_hip_pitch_joint":       0.5475464629911068,
    "left_hip_roll_joint":        0.35066146637882434,
    "left_hip_yaw_joint":         0.5475464629911068,
    "left_knee_joint":            0.35066146637882434,
    "left_ankle_pitch_joint":     0.43857731392336724,
    "left_ankle_roll_joint":      0.43857731392336724,
    "right_hip_pitch_joint":      0.5475464629911068,
    "right_hip_roll_joint":       0.35066146637882434,
    "right_hip_yaw_joint":        0.5475464629911068,
    "right_knee_joint":           0.35066146637882434,
    "right_ankle_pitch_joint":    0.43857731392336724,
    "right_ankle_roll_joint":     0.43857731392336724,
    "waist_yaw_joint":            0.5475464629911068,
    "waist_roll_joint":           0.43857731392336724,
    "waist_pitch_joint":          0.43857731392336724,
    "left_shoulder_pitch_joint":  0.43857731392336724,
    "left_shoulder_roll_joint":   0.43857731392336724,
    "left_shoulder_yaw_joint":    0.43857731392336724,
    "left_elbow_joint":           0.43857731392336724,
    "left_wrist_roll_joint":      0.43857731392336724,
    "left_wrist_pitch_joint":     0.07450087032950714,
    "left_wrist_yaw_joint":       0.07450087032950714,
    "right_shoulder_pitch_joint": 0.43857731392336724,
    "right_shoulder_roll_joint":  0.43857731392336724,
    "right_shoulder_yaw_joint":   0.43857731392336724,
    "right_elbow_joint":          0.43857731392336724,
    "right_wrist_roll_joint":     0.43857731392336724,
    "right_wrist_pitch_joint":    0.07450087032950714,
    "right_wrist_yaw_joint":      0.07450087032950714,
}
_G1_STIFFNESS = {
    "left_hip_pitch_joint":       40.17923863450712,
    "left_hip_roll_joint":        99.09842777666111,
    "left_hip_yaw_joint":         40.17923863450712,
    "left_knee_joint":            99.09842777666111,
    "left_ankle_pitch_joint":     28.50124619574858,
    "left_ankle_roll_joint":      28.50124619574858,
    "right_hip_pitch_joint":      40.17923863450712,
    "right_hip_roll_joint":       99.09842777666111,
    "right_hip_yaw_joint":        40.17923863450712,
    "right_knee_joint":           99.09842777666111,
    "right_ankle_pitch_joint":    28.50124619574858,
    "right_ankle_roll_joint":     28.50124619574858,
    "waist_yaw_joint":            40.17923863450712,
    "waist_roll_joint":           28.50124619574858,
    "waist_pitch_joint":          28.50124619574858,
    "left_shoulder_pitch_joint":  14.25062309787429,
    "left_shoulder_roll_joint":   14.25062309787429,
    "left_shoulder_yaw_joint":    14.25062309787429,
    "left_elbow_joint":           14.25062309787429,
    "left_wrist_roll_joint":      14.25062309787429,
    "left_wrist_pitch_joint":     16.77832748089279,
    "left_wrist_yaw_joint":       16.77832748089279,
    "right_shoulder_pitch_joint": 14.25062309787429,
    "right_shoulder_roll_joint":  14.25062309787429,
    "right_shoulder_yaw_joint":   14.25062309787429,
    "right_elbow_joint":          14.25062309787429,
    "right_wrist_roll_joint":     14.25062309787429,
    "right_wrist_pitch_joint":    16.77832748089279,
    "right_wrist_yaw_joint":      16.77832748089279,
}
_G1_DAMPING = {
    "left_hip_pitch_joint":       2.557889775413375,
    "left_hip_roll_joint":        6.308801853496639,
    "left_hip_yaw_joint":         2.557889775413375,
    "left_knee_joint":            6.308801853496639,
    "left_ankle_pitch_joint":     1.814445686584846,
    "left_ankle_roll_joint":      1.814445686584846,
    "right_hip_pitch_joint":      2.557889775413375,
    "right_hip_roll_joint":       6.308801853496639,
    "right_hip_yaw_joint":        2.557889775413375,
    "right_knee_joint":           6.308801853496639,
    "right_ankle_pitch_joint":    1.814445686584846,
    "right_ankle_roll_joint":     1.814445686584846,
    "waist_yaw_joint":            2.557889775413375,
    "waist_roll_joint":           1.814445686584846,
    "waist_pitch_joint":          1.814445686584846,
    "left_shoulder_pitch_joint":  0.907222843292423,
    "left_shoulder_roll_joint":   0.907222843292423,
    "left_shoulder_yaw_joint":    0.907222843292423,
    "left_elbow_joint":           0.907222843292423,
    "left_wrist_roll_joint":      0.907222843292423,
    "left_wrist_pitch_joint":     1.06814150219,
    "left_wrist_yaw_joint":       1.06814150219,
    "right_shoulder_pitch_joint": 0.907222843292423,
    "right_shoulder_roll_joint":  0.907222843292423,
    "right_shoulder_yaw_joint":   0.907222843292423,
    "right_elbow_joint":          0.907222843292423,
    "right_wrist_roll_joint":     0.907222843292423,
    "right_wrist_pitch_joint":    1.06814150219,
    "right_wrist_yaw_joint":      1.06814150219,
}
# fmt: on


def setup_builder() -> mjswan.Builder:
    """Set up the builder with a splat-backed scene.

    Returns:
        Configured Builder instance ready to be built.
    """
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path)

    project = builder.add_project(name="Splat Demo")

    scene = project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    )

    # G1 uses motor actuators (biastype=none) that need external PD control in
    # the browser runtime, so stiffness/damping are supplied via the action term.
    g1_actions = {
        "joint_pos": JointPositionActionCfg(
            scale=_G1_SCALE,
            stiffness=_G1_STIFFNESS,
            damping=_G1_DAMPING,
        )
    }
    g1_terminations = {
        "bad_orientation": TerminationTermCfg(
            func=term_fns.bad_orientation, params={"limit_angle": 1.0}
        ),
        "root_height_below_minimum": TerminationTermCfg(
            func=term_fns.root_height_below_minimum, params={"minimum_height": 0.3}
        ),
    }

    scene.add_policy(
        name="balance",
        policy=onnx.load("assets/unitree_g1/balance.onnx"),
        config_path="assets/unitree_g1/balance.json",
        actions=g1_actions,
        terminations=g1_terminations,
        observations={
            "observation": ObservationGroupCfg(
                terms={
                    "base_ang_vel": ObservationTermCfg(
                        func=obs_fns.base_ang_vel, history_length=1
                    ),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity,
                        history_length=1,
                        params={"gravity": [0, 0, -1.0]},
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_pos_rel, history_length=1
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel,
                        params={"joint_names": "isaac"},
                        history_length=1,
                    ),
                    "prev_actions": ObservationTermCfg(func=obs_fns.last_action),
                }
            )
        },
    )

    scene.add_policy(
        name="locomotion",
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        config_path="assets/unitree_g1/locomotion.json",
        actions=g1_actions,
        terminations=g1_terminations,
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=obs_fns.joint_pos_rel, params={"pos_steps": [0]}
                    ),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                    "last_action": ObservationTermCfg(func=obs_fns.last_action),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "velocity"},
                    ),
                }
            )
        },
    ).add_velocity_command(
        lin_vel_x=(-1.5, 1.5),
        lin_vel_y=(-0.5, 0.5),
        default_lin_vel_x=0.5,
    )

    for i, splat_url in enumerate(SPLAT_URLs):
        scene.add_splat(
            name=f"Splat {i + 1}",
            url=splat_url,
            scale=3.275,
            z_offset=0.708,
            control=True,
        )

    return builder


def main():
    """Build and launch the splat demo."""
    builder = setup_builder()
    app = builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
