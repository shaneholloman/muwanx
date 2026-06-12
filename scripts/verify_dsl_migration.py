#!/usr/bin/env python
"""Structural verifier for the declarative MDP DSL migration (ADR 0003).

This is the *runnable-without-a-browser* layer of the verification plan
(see docs/dsl-migration-verification.md).  It does NOT prove numeric
equivalence — that needs the vitest harness / browser smoke test — but it
catches the highest-value structural regressions early and fast:

1. **Op parity.** Every primitive op the Python DSL emits for a migrated
   built-in must exist in the engine's `Primitives` registry
   (template/src/core/dsl/primitives.ts).  A mismatch means the engine would
   throw "unknown primitive op" at runtime.

2. **Envelope shape.** Each migrated built-in traces to a well-formed
   envelope (`kind` set; topologically-ordered nodes; resolvable output).

Run::

    uv run python scripts/verify_dsl_migration.py

Exit code 0 = all checks pass; 1 = a discrepancy was found (details printed).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from mjswan.dsl import trace_event, trace_observation, trace_termination
from mjswan.envs.mdp import events as event_fns
from mjswan.envs.mdp import observations as obs_fns
from mjswan.envs.mdp import terminations as term_fns

_ENGINE_PRIMITIVES = (
    Path(__file__).resolve().parent.parent
    / "src/mjswan/template/src/core/dsl/primitives.ts"
)

# Representative params for each migrated built-in.  Mirrors how the mjlab
# adapter / scene enrichment would populate them at build time.
_OBS: dict[str, dict] = {
    "base_lin_vel": {},
    "base_ang_vel": {},
    "projected_gravity": {},
    "joint_pos_rel": {},
    "joint_pos_rel_named": {"joint_names": ["a", "b"], "default_joint_pos": [0.1, 0.2]},
    "joint_vel_rel": {},
    "last_action": {},
    "previous_actions": {},
    "generated_commands": {"command_name": "velocity"},
    "motion_anchor_pos_b": {},
    "motion_anchor_ori_b": {},
    "robot_body_pos_b": {"body_names": ["pelvis", "torso"]},
    "robot_body_ori_b": {"body_names": ["pelvis", "torso"]},
    # ee_to_object_distance / object_to_goal_distance are task-specific terms
    # registered in the task (examples/mjlab/defaults), not core built-ins.
    "builtin_sensor": {"sensor_name": "imu"},
}
_TERM: dict[str, dict] = {
    "time_out": {},
    "bad_orientation": {"limit_angle": 1.0},
    "root_height_below_minimum": {"minimum_height": 0.2},
    "out_of_terrain_bounds": {"limit_x": 2.0, "limit_y": 2.0},
    "terrain_edge_reached": {"half_x": 2.0, "half_y": 2.0},
    "bad_anchor_pos_z_only": {"threshold": 0.5},
    "bad_anchor_ori": {"threshold": 0.5},
    "bad_motion_body_pos_z_only": {"threshold": 0.3},
    "base_ang_vel_exceed": {"threshold": 2.0},
}
_EVENT: dict[str, dict] = {
    "randomize_terrain": {},
    "reset_joints_by_offset": {
        "position_range": [-0.1, 0.1],
        "velocity_range": [-0.1, 0.1],
    },
    "reset_root_state_uniform": {
        "pose_range": {"x": [-1.0, 1.0], "yaw": [-3.14, 3.14]}
    },
}


def _engine_ops() -> set[str]:
    """Extract the engine's primitive-op names from primitives.ts.

    Registry entries are `  Name: (args) => ...` at two-space indent; the
    `: (` arrow-function guard avoids matching helper object keys.
    """
    text = _ENGINE_PRIMITIVES.read_text()
    block = text.split("export const Primitives", 1)[-1]
    return set(re.findall(r"^  (\w+): \(", block, flags=re.MULTILINE))


def _envelope_ops(envelope: dict) -> set[str]:
    if envelope.get("kind") == "event":
        # Events carry mutation descriptors, not graph nodes; no ops to check.
        return set()
    return {n["op"] for n in envelope.get("nodes", [])}


def _check_envelope(name: str, envelope: dict, errors: list[str]) -> None:
    kind = envelope.get("kind")
    if kind not in {"observation", "termination", "event"}:
        errors.append(f"{name}: missing/invalid kind {kind!r}")
        return
    if kind == "event":
        if not isinstance(envelope.get("mutations"), list):
            errors.append(f"{name}: event envelope has no mutations list")
        return
    nodes = envelope.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append(f"{name}: empty/invalid nodes")
        return
    out_names = {n["out"] for n in nodes}
    if envelope.get("output") not in out_names:
        errors.append(f"{name}: output {envelope.get('output')!r} not produced")
    # Topological order: every `in` edge must be defined by an earlier node.
    seen: set[str] = set()
    for node in nodes:
        for src in node.get("in", []):
            if src not in seen:
                errors.append(
                    f"{name}: node {node['out']!r} uses undefined input {src!r}"
                )
        seen.add(node["out"])


def main() -> int:
    engine_ops = _engine_ops()
    if not engine_ops:
        print(f"ERROR: could not parse engine ops from {_ENGINE_PRIMITIVES}")
        return 1
    print(f"Engine primitive ops: {len(engine_ops)}")

    errors: list[str] = []
    emitted: set[str] = set()

    def run(group: dict[str, dict], module, tracer) -> None:
        for label, params in group.items():
            fn_name = label.split("_named")[0] if label.endswith("_named") else label
            func = getattr(module, fn_name)
            envelope = tracer(func, params)
            _check_envelope(label, envelope, errors)
            emitted.update(_envelope_ops(envelope))

    run(_OBS, obs_fns, trace_observation)
    run(_TERM, term_fns, trace_termination)
    run(_EVENT, event_fns, trace_event)

    # Exercise the scale/clip/history baking path (Clip/Mul/History ops).
    baked = trace_observation(
        obs_fns.base_lin_vel,
        {},
        scale=[0.1, 0.2, 0.3],
        clip=(-2.0, 2.0),
        history_steps=3,
    )
    _check_envelope("base_lin_vel+baked", baked, errors)
    emitted.update(_envelope_ops(baked))

    missing = sorted(emitted - engine_ops)
    if missing:
        errors.append(
            f"Ops emitted by Python DSL but absent from engine registry: {missing}"
        )

    print(f"Distinct ops emitted by migrated built-ins: {len(emitted)}")
    unused = sorted(engine_ops - emitted)
    if unused:
        # Informational only — some ops (e.g. Set, scalar helpers) are reachable
        # only via user-authored terms or postproc baking.
        print(f"Engine ops not exercised by built-ins (informational): {unused}")

    if errors:
        print("\nFAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nOK: op parity + envelope shape verified for all migrated built-ins.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
