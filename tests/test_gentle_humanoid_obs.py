"""The Gentle Humanoid tracking observations trace to declarative graphs (#79).

These four terms previously required custom TypeScript because they read the
motion command's private reference buffers.  They are now composition graphs
over the `TrackingRefField` source; this test pins their traced structure and
keeps the shared golden fixture (consumed by the TS equivalence test) current.

Regenerate the fixture after an intentional builder change:
    uv run python -c "import json; from mjswan.dsl import trace_observation; \
from examples.demo.gentle_humanoid.dsl_terms import BUILDERS; \
s=[0,1,2,3,4,-1,-2,-4,-8,-12,-16]; \
json.dump({'future_steps': s, 'graphs': {n: trace_observation(f, {'future_steps': s}) \
for n,f in BUILDERS.items()}}, open('src/mjswan/template/src/core/dsl/__tests__/gentleHumanoidObsGraphs.json','w'), indent=2)"
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


def _golden() -> dict:
    return json.loads(_GOLDEN.read_text())


def test_builders_trace_to_golden_graphs():
    """Traced graphs match the committed fixture the TS test evaluates."""
    golden = _golden()
    steps = golden["future_steps"]
    for name, fn in BUILDERS.items():
        assert trace_observation(fn, {"future_steps": steps}) == golden["graphs"][name]


def test_expected_op_structure_and_sizes():
    """Each term composes the expected ops (and total output width)."""
    steps = [0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]
    n = len(steps)

    def ops(name):
        g = trace_observation(BUILDERS[name], {"future_steps": steps})
        return Counter(node["op"] for node in g["nodes"])

    tracking = ops("gentle_humanoid_tracking")
    # Column-major rot6d is composed from row-major QuatToRot6d + a 6-way
    # reindex, not a dedicated op.
    assert "QuatToRot6dColumns" not in tracking
    assert tracking["QuatToRot6d"] == n and tracking["Index"] == 6 * n
    assert tracking["QuatApplyInv"] == n - 1  # pos deltas skip the base frame
    assert tracking["TrackingIsReady"] == 1 and tracking["Mul"] == 1

    joint = ops("gentle_humanoid_target_joint_pos")
    assert joint["TrackingRefField"] == 2 * n  # targets + diffs
    assert joint["Sub"] == n and joint["JointPos"] == 1

    root_z = ops("gentle_humanoid_target_root_z")
    assert root_z["Index"] == n and root_z["TrackingRefField"] == n

    grav = ops("gentle_humanoid_target_projected_gravity")
    # normalize is composed from Sum/Sqrt/Div, not a dedicated op.
    assert "Normalize" not in grav
    assert grav["Sum"] == n and grav["Sqrt"] == n and grav["Div"] == n
    assert grav["QuatApplyInv"] == n


def test_terms_carry_no_custom_js():
    """The whole point: these are declarative, not ts_src."""
    for fn in BUILDERS.values():
        graph = trace_observation(fn, {"future_steps": [0, 1, -1]})
        assert graph["kind"] == "observation"
        assert "name" not in graph  # a ts_src term would serialize {"name": ...}
