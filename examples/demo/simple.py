"""Simple mjswan Demo

A basic example demonstrating how to use mjswan to create a viewer application
with multiple robot scenes (Go2, Go1, and G1).
"""

import os
from pathlib import Path

import mujoco
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

import mjswan
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.trace_env import build_single_entity_trace_env

# G1 humanoid: per-joint action scale, stiffness, damping (keyed by joint name)
_G1_SCALE = {
    "left_hip_pitch_joint": 0.5475464629911068,
    "left_hip_roll_joint": 0.35066146637882434,
    "left_hip_yaw_joint": 0.5475464629911068,
    "left_knee_joint": 0.35066146637882434,
    "left_ankle_pitch_joint": 0.43857731392336724,
    "left_ankle_roll_joint": 0.43857731392336724,
    "right_hip_pitch_joint": 0.5475464629911068,
    "right_hip_roll_joint": 0.35066146637882434,
    "right_hip_yaw_joint": 0.5475464629911068,
    "right_knee_joint": 0.35066146637882434,
    "right_ankle_pitch_joint": 0.43857731392336724,
    "right_ankle_roll_joint": 0.43857731392336724,
    "waist_yaw_joint": 0.5475464629911068,
    "waist_roll_joint": 0.43857731392336724,
    "waist_pitch_joint": 0.43857731392336724,
    "left_shoulder_pitch_joint": 0.43857731392336724,
    "left_shoulder_roll_joint": 0.43857731392336724,
    "left_shoulder_yaw_joint": 0.43857731392336724,
    "left_elbow_joint": 0.43857731392336724,
    "left_wrist_roll_joint": 0.43857731392336724,
    "left_wrist_pitch_joint": 0.07450087032950714,
    "left_wrist_yaw_joint": 0.07450087032950714,
    "right_shoulder_pitch_joint": 0.43857731392336724,
    "right_shoulder_roll_joint": 0.43857731392336724,
    "right_shoulder_yaw_joint": 0.43857731392336724,
    "right_elbow_joint": 0.43857731392336724,
    "right_wrist_roll_joint": 0.43857731392336724,
    "right_wrist_pitch_joint": 0.07450087032950714,
    "right_wrist_yaw_joint": 0.07450087032950714,
}
_G1_STIFFNESS = {
    "left_hip_pitch_joint": 40.17923863450712,
    "left_hip_roll_joint": 99.09842777666111,
    "left_hip_yaw_joint": 40.17923863450712,
    "left_knee_joint": 99.09842777666111,
    "left_ankle_pitch_joint": 28.50124619574858,
    "left_ankle_roll_joint": 28.50124619574858,
    "right_hip_pitch_joint": 40.17923863450712,
    "right_hip_roll_joint": 99.09842777666111,
    "right_hip_yaw_joint": 40.17923863450712,
    "right_knee_joint": 99.09842777666111,
    "right_ankle_pitch_joint": 28.50124619574858,
    "right_ankle_roll_joint": 28.50124619574858,
    "waist_yaw_joint": 40.17923863450712,
    "waist_roll_joint": 28.50124619574858,
    "waist_pitch_joint": 28.50124619574858,
    "left_shoulder_pitch_joint": 14.25062309787429,
    "left_shoulder_roll_joint": 14.25062309787429,
    "left_shoulder_yaw_joint": 14.25062309787429,
    "left_elbow_joint": 14.25062309787429,
    "left_wrist_roll_joint": 14.25062309787429,
    "left_wrist_pitch_joint": 16.77832748089279,
    "left_wrist_yaw_joint": 16.77832748089279,
    "right_shoulder_pitch_joint": 14.25062309787429,
    "right_shoulder_roll_joint": 14.25062309787429,
    "right_shoulder_yaw_joint": 14.25062309787429,
    "right_elbow_joint": 14.25062309787429,
    "right_wrist_roll_joint": 14.25062309787429,
    "right_wrist_pitch_joint": 16.77832748089279,
    "right_wrist_yaw_joint": 16.77832748089279,
}
_G1_DAMPING = {
    "left_hip_pitch_joint": 2.557889775413375,
    "left_hip_roll_joint": 6.308801853496639,
    "left_hip_yaw_joint": 2.557889775413375,
    "left_knee_joint": 6.308801853496639,
    "left_ankle_pitch_joint": 1.814445686584846,
    "left_ankle_roll_joint": 1.814445686584846,
    "right_hip_pitch_joint": 2.557889775413375,
    "right_hip_roll_joint": 6.308801853496639,
    "right_hip_yaw_joint": 2.557889775413375,
    "right_knee_joint": 6.308801853496639,
    "right_ankle_pitch_joint": 1.814445686584846,
    "right_ankle_roll_joint": 1.814445686584846,
    "waist_yaw_joint": 2.557889775413375,
    "waist_roll_joint": 1.814445686584846,
    "waist_pitch_joint": 1.814445686584846,
    "left_shoulder_pitch_joint": 0.907222843292423,
    "left_shoulder_roll_joint": 0.907222843292423,
    "left_shoulder_yaw_joint": 0.907222843292423,
    "left_elbow_joint": 0.907222843292423,
    "left_wrist_roll_joint": 0.907222843292423,
    "left_wrist_pitch_joint": 1.06814150219,
    "left_wrist_yaw_joint": 1.06814150219,
    "right_shoulder_pitch_joint": 0.907222843292423,
    "right_shoulder_roll_joint": 0.907222843292423,
    "right_shoulder_yaw_joint": 0.907222843292423,
    "right_elbow_joint": 0.907222843292423,
    "right_wrist_roll_joint": 0.907222843292423,
    "right_wrist_pitch_joint": 1.06814150219,
    "right_wrist_yaw_joint": 1.06814150219,
}
_G1_OFFSET = {
    "left_shoulder_pitch_joint": 0.5,
    "right_shoulder_pitch_joint": -0.5,
    "left_wrist_roll_joint": 2.0,
    "right_wrist_roll_joint": -2.0,
}


def setup_builder() -> mjswan.Builder:
    """Set up and return the builder with demo projects configured.

    Creates a builder and adds a project with three robot scenes.
    Does not build or launch the application.

    Returns:
        Configured Builder instance ready to be built.
    """
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path)

    demo_project = builder.add_project(
        name="mjswan Demo",
    )

    demo_project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    ).set_trace_env(
        # The env the policy's observation terms are traced against (ADR 0005 §6).
        # An mjlab scene brings its own; this one is a plain MJCF, so it needs the
        # entity built explicitly — from the robot alone, not the scene's floor.
        build_single_entity_trace_env(
            lambda: mujoco.MjSpec.from_file("assets/unitree_g1/g1.xml")
        )
    ).set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.0),
            distance=2.5,
            elevation=-10.0,
            azimuth=-34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="torso_link",
        )
    ).add_policy(
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        name="Locomotion",
        config_path="assets/unitree_g1/locomotion.json",
        actions={
            # mjlab's JointPositionActionCfg has no stiffness/damping fields, so
            # mjswan's JointPositionActionCfg is used directly for the actions dict.
            # Stiffness/damping are required here because G1 uses motor actuators
            # (biastype=none) that need external PD control in the browser runtime.
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=_G1_SCALE,
                offset=_G1_OFFSET,
                stiffness=_G1_STIFFNESS,
                damping=_G1_DAMPING,
            ),
        },
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                    "last_action": ObservationTermCfg(func=obs_fns.last_action),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "velocity"},
                    ),
                }
            )
        },
        commands={
            "velocity": mjswan.velocity_command(
                lin_vel_x=(-2.0, 2.0),
                lin_vel_y=(-0.5, 0.5),
                default_lin_vel_x=0.5,
                default_lin_vel_y=0.0,
            )
        },
    )
    demo_project.add_scene(
        # model=mujoco.MjModel.from_xml_path("assets/unitree_go2/scene.xml"),
        spec=mujoco.MjSpec.from_file("assets/unitree_go2/scene.xml"),
        name="Go2",
    )

    return builder


def main():
    """Main entry point for the simple demo.

    Sets up the builder, builds the application, and launches it in a browser.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    builder = setup_builder()
    # Build and launch the application
    app = builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
