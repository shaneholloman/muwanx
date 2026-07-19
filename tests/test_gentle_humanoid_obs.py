"""The Gentle Humanoid observations trace to declarative graphs (#79).

All eleven obs are now composition graphs (Slice B: the four command-coupled
terms; Slice C: boot / compliance / the four proprioceptive histories /
prev_actions).  This pins their traced structure and keeps the shared golden
fixture (consumed by the TS byte-equivalence test) current.

Regenerate the fixture after an intentional builder change:
    uv run python -m tests.test_gentle_humanoid_obs
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from examples.demo.gentle_humanoid.dsl_terms import BUILDERS
from mjswan.dsl import trace_observation

_GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "src/mjswan/template/src/core/dsl/__tests__/gentleHumanoidObsGraphs.json"
)

FUTURE_STEPS = [0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]
HISTORY_STEPS = [0, 1, 2, 3, 4, 8, 12, 16, 20]
NUM_JOINTS = 5

# Trace-time params per obs (a small NUM_JOINTS keeps the fixture readable).
PARAMS: dict[str, dict] = {
    "gentle_humanoid_tracking": {"future_steps": FUTURE_STEPS},
    "gentle_humanoid_target_joint_pos": {
        "future_steps": FUTURE_STEPS,
        "num_joints": NUM_JOINTS,
    },
    "gentle_humanoid_target_root_z": {"future_steps": FUTURE_STEPS},
    "gentle_humanoid_target_projected_gravity": {"future_steps": FUTURE_STEPS},
    "gentle_humanoid_boot": {},
    "gentle_humanoid_compliance": {"command_name": "compliance"},
    "gentle_humanoid_root_ang_vel": {"history_steps": HISTORY_STEPS},
    "gentle_humanoid_projected_gravity": {"history_steps": HISTORY_STEPS},
    "gentle_humanoid_joint_pos": {
        "history_steps": HISTORY_STEPS,
        "num_joints": NUM_JOINTS,
    },
    "gentle_humanoid_joint_vel": {
        "history_steps": HISTORY_STEPS,
        "num_joints": NUM_JOINTS,
    },
    "gentle_humanoid_prev_actions": {"history_steps": 8},
}


def _build_graphs() -> dict:
    graphs = {
        name: trace_observation(BUILDERS[name], PARAMS[name]) for name in BUILDERS
    }
    return {
        "future_steps": FUTURE_STEPS,
        "history_steps": HISTORY_STEPS,
        "num_joints": NUM_JOINTS,
        "prev_action_steps": PARAMS["gentle_humanoid_prev_actions"]["history_steps"],
        "graphs": graphs,
    }


def _golden() -> dict:
    return json.loads(_GOLDEN.read_text())


def test_all_eleven_obs_have_params():
    assert set(BUILDERS) == set(PARAMS)


def test_builders_trace_to_golden_graphs():
    """Traced graphs match the committed fixture the TS test evaluates."""
    assert _build_graphs()["graphs"] == _golden()["graphs"]


def test_slice_c_op_structure():
    def ops(name):
        return Counter(
            n["op"] for n in trace_observation(BUILDERS[name], PARAMS[name])["nodes"]
        )

    assert ops("gentle_humanoid_boot") == Counter({"ConstVec": 1})

    compliance = ops("gentle_humanoid_compliance")
    assert (
        compliance["CommandValue"] == 1
        and compliance["Ge"] == 1
        and compliance["Div"] == 1
    )

    n = len(HISTORY_STEPS)
    for name in ("root_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
        o = ops(f"gentle_humanoid_{name}")
        # sparse selection: one History stack, one Slice per look-back offset.
        assert o["History"] == 1 and o["Slice"] == n and o["Concat"] == 1

    prev = ops("gentle_humanoid_prev_actions")
    assert prev["PrevAction"] == 1 and prev["History"] == 1 and "Slice" not in prev


def test_terms_carry_no_custom_js():
    """All eleven are declarative, not ts_src."""
    for name, params in PARAMS.items():
        graph = trace_observation(BUILDERS[name], params)
        assert graph["kind"] == "observation" and "name" not in graph


def _regenerate() -> None:
    _GOLDEN.write_text(json.dumps(_build_graphs(), indent=2) + "\n")
    print(f"wrote {_GOLDEN}")


if __name__ == "__main__":
    _regenerate()
