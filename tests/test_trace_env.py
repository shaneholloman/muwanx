"""The trace env's defaults, which every `*_rel` observation bakes in.

Layer: L3 (builds a real mjlab env from a spec).

`build_single_entity_trace_env` exists so a plain `add_scene()` scene can be
traced at all. What it hands the tracer is not just kinematics: `joint_pos_rel`
and friends bake `default_joint_pos` into the graph as a constant, so a wrong
default is not a wrong *reading* — it is a permanent offset in what the policy
sees, invisible in the artifact and fatal in the browser (a balance policy reads
its whole stand pose as error and falls over).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")

import mujoco  # noqa: E402

from mjswan.trace_env import build_single_entity_trace_env  # noqa: E402

STAND = {"hinge_a": 0.25, "hinge_b": -0.4}

MODEL = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="link_a" pos="0 0 0.2">
        <joint name="hinge_a" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.1"/>
        <body name="link_b" pos="0 0 0.2">
          <joint name="hinge_b" type="hinge" axis="0 1 0"/>
          <geom type="box" size="0.05 0.05 0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  {keyframe}
</mujoco>
"""

KEYFRAME = f"""
  <keyframe>
    <key name="stand" qpos="0 0 1 1 0 0 0 {STAND["hinge_a"]} {STAND["hinge_b"]}"/>
  </keyframe>
"""


def _spec_fn(keyframe: str):
    return lambda: mujoco.MjSpec.from_string(MODEL.format(keyframe=keyframe))


def test_default_joint_pos_comes_from_the_models_keyframe():
    """The browser resets to keyframe 0, so the traced default has to be that pose."""
    env = build_single_entity_trace_env(_spec_fn(KEYFRAME))
    defaults = env.scene["robot"].data.default_joint_pos.reshape(-1).tolist()
    names = env.scene["robot"].joint_names
    assert dict(zip(names, defaults)) == pytest.approx(STAND)


def test_a_model_without_a_keyframe_keeps_mjlabs_zero_default():
    """No keyframe means the zero pose *is* the rest pose — mjlab's own default."""
    env = build_single_entity_trace_env(_spec_fn(""))
    defaults = env.scene["robot"].data.default_joint_pos.reshape(-1).tolist()
    assert defaults == [0.0, 0.0]
