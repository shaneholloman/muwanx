"""Dump a slot-reader parity fixture from a live mjlab task.

The TypeScript slot reader (``core/onnx/slotReader.ts``) reimplements mjlab's
``EntityData`` field semantics against raw ``mjModel``/``mjData``. Nothing in the
Python parity harness covers that reimplementation: the harness proves each
traced *graph* matches mjlab, while the reader decides what numbers go *into* the
graph, browser-side. A wrong address or element order there produces a policy
that runs happily on the wrong state.

So: step a real task, write out the model/data arrays the reader indexes plus
mjlab's own value for every field it claims to serve, and let a vitest case assert
the two agree. Regenerate with::

    MUJOCO_GL=disable .venv/bin/python scripts/dump_slot_fixture.py

Ground truth is read off ``env.scene[entity].data`` — the same property the tracer
recorded — so the fixture stays honest even if mjlab changes a definition.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "disable")

OUT = (
    Path(__file__).resolve().parents[1]
    / "src/mjswan/template/src/core/onnx/__tests__/fixtures/slotFields.json"
)

# One task with prefixed entities, sites and a free joint; one with sensors and 29 joints.
TASKS = ("Mjlab-Lift-Cube-Yam", "Mjlab-Velocity-Flat-Unitree-G1")

# Every field the reader implements, so a newly-added one cannot skip the check.
FIELDS = (
    "joint_pos",
    "joint_pos_biased",
    "joint_vel",
    "root_link_pos_w",
    "root_link_quat_w",
    "root_link_pose_w",
    "root_link_vel_w",
    "root_link_lin_vel_w",
    "root_link_ang_vel_w",
    "root_link_lin_vel_b",
    "root_link_ang_vel_b",
    "projected_gravity_b",
    "heading_w",
    "site_pos_w",
)

MODEL_INTS = (
    "njnt",
    "nbody",
    "nsite",
    "nsensor",
)
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
# mjData fields, each sliced to env 0 (the browser runs a single env).
DATA_ARRAYS = (
    "qpos",
    "qvel",
    "xpos",
    "xquat",
    "cvel",
    "subtree_com",
    "site_xpos",
    "sensordata",
)


def _flat(value: Any) -> list[float]:
    return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]


def _dump_task(task_id: str) -> dict[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task_id, play=True)
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg, device="cpu")
        env.reset()
        # A few steps, so a reader that just returned qpos0 cannot pass at t=0.
        for _ in range(3):
            env.sim.forward()
            env.scene.update(env.step_dt)

    mj_model = env.sim.mj_model
    model = {name: int(getattr(mj_model, name)) for name in MODEL_INTS}
    for name in MODEL_ARRAYS:
        model[name] = [int(v) for v in getattr(mj_model, name).reshape(-1).tolist()]
    # The NUL-separated name blob ships as a list of ints, to keep the fixture plain JSON.
    model["names"] = list(bytes(mj_model.names))

    data = {name: _flat(getattr(env.sim.data, name)[0]) for name in DATA_ARRAYS}

    entities: dict[str, dict[str, Any]] = {}
    for entity_name in env.scene.entities:
        entity = env.scene[entity_name]
        entity_data = entity.data
        fields: dict[str, list[float]] = {}
        for field in FIELDS:
            try:
                value = getattr(entity_data, field)
            except Exception:  # noqa: BLE001 — a field this entity cannot report
                continue
            fields[field] = _flat(value[0] if value.ndim > 0 else value)
        # The walking tasks randomize `encoder_bias`, so the reader needs the same bias.
        bias = _flat(entity_data.encoder_bias[0])
        entities[entity_name] = {
            "fields": fields,
            "encoder_bias": dict(zip(entity.joint_names, bias, strict=True)),
        }

    # Only builtin sensors are `sensordata` windows; the tracer already rejects the rest.
    sensors: dict[str, list[float]] = {}
    for name, sensor in env.scene.sensors.items():
        value = sensor.data
        if hasattr(value, "ndim") and hasattr(value, "detach"):
            sensors[name] = _flat(value[0])

    return {"model": model, "data": data, "entities": entities, "sensors": sensors}


def main() -> None:
    fixture = {task_id: _dump_task(task_id) for task_id in TASKS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    for task_id, payload in fixture.items():
        entities = payload["entities"]
        print(
            f"{task_id}: {len(entities)} entities "
            f"({', '.join(sorted(entities))}), "
            f"{len(payload['sensors'])} sensors"
        )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
