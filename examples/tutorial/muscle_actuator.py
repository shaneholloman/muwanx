"""Muscle Actuator Policy

Validates three features added for muscle-driven policies:

1. ``policy_num_actions`` — declare action count without ``policy_joint_names``
2. ``initial_qpos`` / ``initial_qvel`` — override keyframe state on reset
3. ``MuscleActivationActionCfg`` — maps outputs through sigmoid(5*(a−0.5))

The model is a single-joint arm (elbow hinge) with two MuJoCo muscle actuators
(bicep, tricep) attached via spatial tendons — matching the pattern in the
fullbody MyoSuite models.  A 2→2 ONNX policy outputs constant 0.5 activations
so both muscles fire at 50%, holding the arm in place.  On every reset the arm
starts at −1 rad (≈−57°, elbow bent) rather than the keyframe default of 0 rad.
"""

import mujoco
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import mjswan
from mjswan.envs.mdp import observations as obs_fns
from mjswan.envs.mdp.actions import MuscleActivationActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

NUM_MUSCLES = 2
OBS_DIM = (
    4  # JointPos(2) + JointVelocities(2), each using policy_num_actions as joint count
)
INITIAL_QPOS = [-1.0]  # elbow bent at -1 rad instead of keyframe 0
INITIAL_QVEL = [0.0]


def build_policy() -> onnx.ModelProto:
    """2→2 policy: zero weights + bias 0.5 → constant 50% muscle activation."""
    W = np.zeros((OBS_DIM, NUM_MUSCLES), dtype=np.float32)
    b = np.full((NUM_MUSCLES,), 0.5, dtype=np.float32)

    obs_in = helper.make_tensor_value_info("policy", TensorProto.FLOAT, [1, OBS_DIM])
    act_out = helper.make_tensor_value_info(
        "action", TensorProto.FLOAT, [1, NUM_MUSCLES]
    )

    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["policy", "W"], ["linear"]),
            helper.make_node("Add", ["linear", "b"], ["action"]),
        ],
        name="muscle_policy",
        inputs=[obs_in],
        outputs=[act_out],
        initializer=[
            numpy_helper.from_array(W, name="W"),
            numpy_helper.from_array(b, name="b"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="mjswan-tutorial",
    )
    onnx.checker.check_model(model)
    return model


_MUSCLE_PARAMS = (
    "ctrllimited='true' ctrlrange='0 1'"
    " dyntype='muscle' dynprm='0.01 0.04 0 0 0 0 0 0 0 0'"
    " gaintype='muscle' biastype='muscle'"
    " gainprm='0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2'"
    " biasprm='0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2'"
    " lengthrange='0.05 0.8'"
)


def build_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(f"""
    <mujoco model="muscle_arm">
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light diffuse=".8 .8 .8" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .92 1"/>
        <body name="upper_arm" pos="0 0 1.2">
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba=".7 .4 .2 1"/>
          <!-- Bicep origin: anterior side of upper arm -->
          <site name="bic_orig" pos="0.01 0.04 -0.05" size="0.008"/>
          <!-- Tricep origin: posterior side of upper arm -->
          <site name="tri_orig" pos="-0.005 -0.035 -0.08" size="0.008"/>
          <body name="forearm" pos="0 0 -0.3">
            <joint name="elbow" type="hinge" axis="1 0 0" range="-2.09 0.17"
                   armature="0.0001" damping="0.75"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025" rgba=".5 .7 .3 1"/>
            <!-- Bicep insertion: anterior side (radial tuberosity) -->
            <site name="bic_ins" pos="0.01 0.03 -0.06" size="0.008"/>
            <!-- Tricep insertion: olecranon (posterior, proximal) -->
            <site name="tri_ins" pos="-0.005 -0.02 0.01" size="0.008"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="bicep_tendon" width="0.006" rgba="0.95 0.3 0.3 1">
          <site site="bic_orig"/>
          <site site="bic_ins"/>
        </spatial>
        <spatial name="tricep_tendon" width="0.006" rgba="0.95 0.3 0.3 1">
          <site site="tri_orig"/>
          <site site="tri_ins"/>
        </spatial>
      </tendon>
      <actuator>
        <general name="bicep"  tendon="bicep_tendon"  {_MUSCLE_PARAMS}/>
        <general name="tricep" tendon="tricep_tendon" {_MUSCLE_PARAMS}/>
      </actuator>
      <keyframe>
        <key name="rest" qpos="0"/>
      </keyframe>
    </mujoco>
    """)


def main():
    builder = mjswan.Builder(debug=True)
    project = builder.add_project(name="Muscle Actuator")

    scene = project.add_scene(spec=build_spec(), name="Simple Arm")

    handles = scene.add_policy(
        name="Muscle Policy",
        policy=build_policy(),
        policy_joint_names=[],  # muscle policy: no joint-name mapping
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "elbow_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                    "elbow_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                }
            ),
        },
        actions={
            "muscles": MuscleActivationActionCfg(
                entity_name="",
                actuator_names=("bicep", "tricep"),
            ),
        },
    )

    handle = handles
    handle._config.policy_num_actions = NUM_MUSCLES  # feature 1
    handle._config.initial_qpos = INITIAL_QPOS  # feature 2
    handle._config.initial_qvel = INITIAL_QVEL

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
