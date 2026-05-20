"""Muscle Actuator Demo.

Showcases mjswan's muscle-driven policy support using MyoFinger from MyoHub:
https://github.com/MyoHub/myo_sim

The MyoFinger model has 4 hinge joints (IFadb, IFmcp, IFpip, IFdip) driven
by 5 MuJoCo muscle actuators via ``MuscleActivationActionCfg``.  A trivial
8 -> 5 ONNX policy outputs constant 0.5 activations so every muscle fires
at 50%.

This demo exercises three features used by muscle-driven policies:

1. ``policy_num_actions`` -- declare action count without ``policy_joint_names``.
2. ``initial_qpos`` / ``initial_qvel`` -- override the rest pose on reset.
3. ``MuscleActivationActionCfg`` -- maps outputs through ``sigmoid(5*(a-0.5))``.

MyoFinger XMLs are fetched at runtime from upstream, so this example adds no
Python dependency on ``myo_sim``.
"""

from pathlib import Path
from urllib.request import urlretrieve

import mujoco
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import mjswan
from mjswan.envs.mdp import observations as obs_fns
from mjswan.envs.mdp.actions import MuscleActivationActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

NUM_MUSCLES = 5  # extn, adabR, adabL, mflx, dflx
NUM_JOINTS = 4  # IFadb, IFmcp, IFpip, IFdip
OBS_DIM = 2 * NUM_JOINTS  # joint_pos + joint_vel

# Slightly flexed finger instead of the model's fully extended default.
INITIAL_QPOS = [0.0, 0.3, 0.3, 0.3]
INITIAL_QVEL = [0.0] * NUM_JOINTS

_MYOFINGER_BASE = "https://raw.githubusercontent.com/MyoHub/myo_sim/main/finger"
_CACHE_DIR = Path.home() / ".cache" / "mjswan" / "myofinger"


def _fetch_myofinger() -> Path:
    """Download MyoFinger XMLs into the user cache; return the entry-point XML."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("myofinger_v0.xml", "finger_v0.xml"):
        target = _CACHE_DIR / name
        if not target.exists():
            urlretrieve(f"{_MYOFINGER_BASE}/{name}", target)
    return _CACHE_DIR / "myofinger_v0.xml"


def _build_policy() -> onnx.ModelProto:
    """8 -> 5 policy with zero weights and bias 0.5 (constant 50% activation)."""
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
        producer_name="mjswan-demo",
    )
    onnx.checker.check_model(model)
    return model


def setup_builder() -> mjswan.Builder:
    builder = mjswan.Builder()
    project = builder.add_project(name="Muscle Actuator")

    scene = project.add_scene(
        spec=mujoco.MjSpec.from_file(str(_fetch_myofinger())),
        name="MyoFinger",
    )

    handle = scene.add_policy(
        name="Constant 50% Activation",
        policy=_build_policy(),
        policy_joint_names=[],  # muscle policy: no joint-name mapping
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                }
            ),
        },
        actions={
            "muscles": MuscleActivationActionCfg(
                entity_name="",
                actuator_names=("extn", "adabR", "adabL", "mflx", "dflx"),
            ),
        },
    )

    handle._config.policy_num_actions = NUM_MUSCLES
    handle._config.initial_qpos = INITIAL_QPOS
    handle._config.initial_qvel = INITIAL_QVEL

    return builder


def main():
    builder = setup_builder()
    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
