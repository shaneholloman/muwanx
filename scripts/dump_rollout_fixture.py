"""Dump an N-step rollout-parity fixture from live mjlab tasks.

ADR 0005's first acceptance criterion is an N-step rollout whose observations and
termination flags match the mjlab reference. Everything shipped so far verifies a
*piece*: the Python harness proves each traced graph reproduces its mjlab term,
``slotReaderParity`` proves the reader feeds those graphs mjlab's numbers, and the
manager unit tests prove the pipeline arithmetic. Nothing checked the whole chain
composed — reader → fused graph → clip/scale → concat → group vector, and reader →
termination graph → lanes → OR-reduce — which is where a layout offset or a
misordered lane would hide.

**Why states are replayed rather than co-simulated.** mjlab integrates with
``mujoco_warp``; the browser runs MuJoCo's own WASM build. Two different
integrators do not agree step-for-step, so a free-running trajectory comparison
would measure MuJoCo against itself, not mjswan against mjlab. Instead each step's
mjlab state is captured and replayed: the fixture carries the ``mjModel``/``mjData``
arrays the reader indexes at that step, and mjlab's own observation vector and
termination verdicts *at that same state*. That isolates exactly the layer mjswan
owns. (Physics is MuJoCo's own code on both sides and is out of scope for this
comparison.)

Termination verdicts are evaluated fresh at the recorded state rather than taken
from ``env.step()``'s return, because mjlab computes terminations *before* its
single ``forward()`` and therefore on derived quantities one substep stale. That
staleness is a step-loop concern (the runtime reproduces it — see the §8 row in the
companion brief), not a question about whether the graph matches the term.

Only the group mjswan actually ships is dumped (mjlab's ``actor``, which the
examples map to ``policy``); ``critic`` is training-only and never reaches a bundle.

Regenerate with::

    MUJOCO_GL=disable .venv/bin/python scripts/dump_rollout_fixture.py
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "disable")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src/mjswan/template/src/core/engine/__tests__/fixtures/rollout"

# Cartpole is ADR 0005's named acceptance criterion and the minimal shape (two
# entity-data slots, one native termination). G1-Velocity-Flat is the wide one:
# builtin-sensor slots, `joint_pos_biased` (so the randomized encoder bias has to
# travel), seven terms in one 99-wide fused group, native `prev_action`/`command`
# inputs, and a traced `fell_over` termination beside the native `time_out`.
#
# Deliberately excluded, each already covered on its own: Velocity-Rough adds a
# `height_scan` raycast slot (`raycast.test.ts`, against the real WASM) and
# Lift-Cube-Yam adds a command-state slot needing a live `OnnxCommand`
# (`OnnxCommand` tests). Both need a stub the fixture format does not carry yet.
TASKS = ("Mjlab-Cartpole-Balance", "Mjlab-Velocity-Flat-Unitree-G1")

STEPS = 20

# `mjModel` fields the slot reader indexes (mirrors `dump_slot_fixture.py`).
MODEL_INTS = ("njnt", "nbody", "nsite", "nsensor")
MODEL_ARRAYS = (
    "jnt_type",
    "jnt_qposadr",
    "jnt_dofadr",
    "name_jntadr",
    "name_bodyadr",
    "name_siteadr",
    "name_sensoradr",
    "sensor_adr",
    "sensor_dim",
)
# `mjData` fields, sliced to env 0 (the browser runs a single env), each paired
# with the `mjModel` count that decides whether the model has any of that element.
# The count is consulted rather than caught: mjlab wraps warp arrays for torch, and
# warp cannot give a torch dtype for a *zero-length vector* array — Cartpole has no
# sites, so reading `site_xpos` raises instead of returning empty.
DATA_ARRAYS = (
    ("qpos", "nq"),
    ("qvel", "nv"),
    ("xpos", "nbody"),
    ("xquat", "nbody"),
    ("cvel", "nbody"),
    ("subtree_com", "nbody"),
    ("site_xpos", "nsite"),
    ("sensordata", "nsensordata"),
)


def _flat(value: Any) -> list[float]:
    return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]


def _data_field(env: Any, name: str, count_attr: str) -> list[float]:
    """One ``mjData`` array for env 0, or empty when the model has no such element."""
    if int(getattr(env.sim.mj_model, count_attr)) == 0:
        return []
    return _flat(getattr(env.sim.data, name)[0])


def _action(step: int, num_actions: int) -> list[float]:
    """A deterministic action sequence.

    A formula rather than a seeded draw so the sequence is identical whatever
    torch does, and bounded so the robot neither freezes nor flails off the map —
    the states have to stay in the region the observation terms are meaningful in.
    """
    return [0.4 * math.sin(0.35 * step + 0.17 * j) for j in range(num_actions)]


def _termination_verdicts(env: Any, cfg: Any) -> dict[str, bool]:
    """Each termination term's verdict at the *current* state.

    Re-evaluated rather than read off `termination_manager`: that one ran before
    mjlab's single `forward()`, so its derived quantities are a substep stale. What
    is under test here is whether the graph agrees with the term, so both sides
    must see the same state.
    """
    from mjswan._onnx_build import _resolved_params

    verdicts: dict[str, bool] = {}
    for name, term_cfg in cfg.terminations.items():
        params = _resolved_params(term_cfg.params, env)
        value = term_cfg.func(env, **params)
        verdicts[name] = bool(value.reshape(-1)[0].item())
    return verdicts


def _native_inputs(env: Any, group_entry: dict[str, Any]) -> dict[str, list[float]]:
    """The values the orchestrator supplies natively, not through a graph.

    `prev_action` and `command` are computed browser-side (the policy's own last
    output; a live command term), so feeding mjlab's value here keeps the
    comparison about the graph and the pipeline. Both are covered separately by
    the `PolicyRunner` and `OnnxCommand` suites.
    """
    natives: dict[str, list[float]] = {}
    for native in group_entry.get("native_inputs", []):
        kind = native["native"]
        if kind == "prev_action":
            natives[native["input"]] = _flat(env.action_manager.action)
        elif kind == "command":
            natives[native["input"]] = _flat(
                env.command_manager.get_command(native["command_name"])
            )
        else:  # pragma: no cover — a new native marker needs a decision here
            raise ValueError(f"no fixture source for native input {kind!r}")
    return natives


# Root pitches to sweep, in radians. Chosen to bracket an orientation limit from
# both sides and to come back under it: a term that latched once it fired, or one
# whose threshold sat in the wrong place, would show up as the wrong steps firing.
# mjlab's `bad_orientation` compares `projected_gravity_b`, so pitching the root is
# the shortest route to flipping it.
TILT_PITCHES = (0.0, 0.4, 1.2, 1.6, 2.2, 2.8, 0.2)


def _append_tilt_sweep(env: Any, cfg: Any, record: Any, num_actions: int) -> None:
    """Force the root orientation through a range, recording each state.

    Only for a floating base: the root quaternion lives at ``qpos[3:7]`` behind a
    free joint. A task without one (Cartpole) has no orientation term to trip, so
    there is nothing to add.

    The pose is written straight into ``qpos`` rather than reached by stepping —
    walking a robot into a fall takes a controller, and what needs covering here is
    the term's verdict at a tilted state, not how it got there.
    """
    import mujoco
    import torch

    mj_model = env.sim.mj_model
    if mj_model.njnt == 0 or mj_model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        return

    for pitch in TILT_PITCHES:
        half = pitch / 2.0
        # Quaternion for a pitch about the body y-axis, mjlab's (w, x, y, z) order.
        quat = [math.cos(half), 0.0, math.sin(half), 0.0]
        qpos = env.sim.data.qpos
        qpos[0, 3:7] = torch.tensor(quat, dtype=qpos.dtype)
        env.sim.forward()
        env.scene.update(env.step_dt)
        record(_action(0, num_actions))


def _dump_task(task_id: str, out_dir: Path) -> dict[str, Any]:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjswan._onnx_build import serialize_observation_group, serialize_terminations
    from mjswan.adapters.mjlab_adapter import _adapt_obs_group, _adapt_term_cfg

    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = 1
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg, device="cpu")
        env.reset()

    # The same serializers the Builder runs, writing the same graph bytes a bundle
    # would carry — so the test loads what really ships, not a stand-in.
    group_entry = serialize_observation_group(
        _adapt_obs_group(cfg.observations["actor"]), env, out_dir, group_name="policy"
    )
    if not isinstance(group_entry, dict) or "fused" not in group_entry:
        raise RuntimeError(
            f"{task_id}: the actor group did not fuse; this fixture only covers "
            "the fused path."
        )
    termination_entries = serialize_terminations(
        {name: _adapt_term_cfg(term) for name, term in cfg.terminations.items()},
        env,
        out_dir,
    )

    mj_model = env.sim.mj_model
    model: dict[str, Any] = {name: int(getattr(mj_model, name)) for name in MODEL_INTS}
    for name in MODEL_ARRAYS:
        model[name] = [int(v) for v in getattr(mj_model, name).reshape(-1).tolist()]
    model["names"] = list(bytes(mj_model.names))

    num_actions = env.action_manager.total_action_dim
    steps: list[dict[str, Any]] = []

    def record(action: list[float]) -> None:
        steps.append(
            {
                "action": action,
                "data": {
                    name: _data_field(env, name, count) for name, count in DATA_ARRAYS
                },
                "native": _native_inputs(env, group_entry),
                "obs": _flat(env.observation_manager.compute_group("actor")),
                "terminations": _termination_verdicts(env, cfg),
            }
        )

    for step in range(STEPS):
        action = _action(step, num_actions)
        env.step(torch.tensor([action], dtype=torch.float32))
        record(action)

    # A natural rollout keeps the robot upright, so every termination reads False
    # and the comparison would only prove the graph raises no false positives — a
    # graph hardwired to False would pass it. Tilting the root through the limit
    # makes an orientation term fire, so both polarities are covered.
    _append_tilt_sweep(env, cfg, record, num_actions)

    # `joint_pos_biased` observes `joint_pos + encoder_bias`, and the walking tasks
    # randomize that bias at startup — a bundle ships it in `policy.json`, so the
    # reader has to be handed the same lookup. Keyed by mjlab's unprefixed joint
    # name, which is unique across a scene.
    encoder_bias: dict[str, float] = {}
    for entity_name in env.scene.entities:
        entity = env.scene[entity_name]
        with contextlib.suppress(AttributeError):
            bias = _flat(entity.data.encoder_bias[0])
            encoder_bias.update(zip(entity.joint_names, bias, strict=True))

    payload = {
        "group": group_entry,
        "terminations": termination_entries,
        "model": model,
        "encoder_bias": encoder_bias,
        "num_actions": num_actions,
        "steps": steps,
    }
    env.close()
    return payload


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    fixture: dict[str, Any] = {}
    for task_id in TASKS:
        graph_dir = OUT_DIR / task_id
        graph_dir.mkdir(parents=True, exist_ok=True)
        fixture[task_id] = _dump_task(task_id, graph_dir)

    index = OUT_DIR / "rollout.json"
    index.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")

    for task_id, payload in fixture.items():
        graphs = sorted(
            p.relative_to(OUT_DIR).as_posix()
            for p in (OUT_DIR / task_id).rglob("*.onnx")
        )
        traced = [n for n, e in payload["terminations"].items() if "onnx" in e]
        print(
            f"{task_id}: {len(payload['steps'])} steps, "
            f"obs width {payload['group']['size']}, "
            f"{len(payload['group']['input_slots'])} slots + "
            f"{len(payload['group']['native_inputs'])} native, "
            f"{len(payload['terminations'])} terminations "
            f"({len(traced)} traced), graphs: {', '.join(graphs)}"
        )
    print(f"wrote {index} ({index.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
