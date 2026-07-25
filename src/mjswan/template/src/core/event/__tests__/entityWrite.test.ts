/**
 * `entity_write` apply primitive (companion brief §3).
 *
 * The half `DslEvent` never had: take a value an ONNX graph already computed and
 * write it into mjData. The write kinds and the `"<kind>__<field>"` output naming
 * mirror the Python tracer's `_WRITE_FIELDS`, so these tests double as the
 * contract between the two sides.
 */
import { describe, expect, it } from 'vitest';

import { applyEntityWrite, applyEntityWrites, findFreeJoint } from '../entityWrite';
import type { WriteTarget } from '../entityWrite';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** Free joint (7 qpos / 6 dof) followed by `nHinge` hinges. */
function fakeModel(nHinge: number, withFreeJoint = true): MjModel {
  const jntType: number[] = [];
  const qposadr: number[] = [];
  const dofadr: number[] = [];
  let q = 0;
  let d = 0;
  if (withFreeJoint) {
    jntType.push(0); // mjJNT_FREE
    qposadr.push(q);
    dofadr.push(d);
    q += 7;
    d += 6;
  }
  for (let i = 0; i < nHinge; i++) {
    jntType.push(3); // mjJNT_HINGE
    qposadr.push(q);
    dofadr.push(d);
    q += 1;
    d += 1;
  }
  return {
    njnt: jntType.length,
    jnt_type: Int32Array.from(jntType),
    jnt_qposadr: Int32Array.from(qposadr),
    jnt_dofadr: Int32Array.from(dofadr),
    names: new Uint8Array(0).buffer,
    name_jntadr: Int32Array.from(jntType.map(() => 0)),
    _nq: q,
    _nv: d,
  } as unknown as MjModel;
}

function fakeData(model: MjModel): MjData {
  const m = model as unknown as { _nq: number; _nv: number };
  return {
    qpos: new Float64Array(m._nq),
    qvel: new Float64Array(m._nv),
  } as unknown as MjData;
}

describe('findFreeJoint', () => {
  it('finds the free joint', () => {
    expect(findFreeJoint(fakeModel(2))).toBe(0);
  });

  it('returns -1 for a fixed-base model', () => {
    expect(findFreeJoint(fakeModel(2, false))).toBe(-1);
  });
});

describe('applyEntityWrite: joint_state', () => {
  it('writes qpos/qvel at the resolved joint ids', () => {
    const model = fakeModel(3, false);
    const data = fakeData(model);
    const target: WriteTarget = {
      kind: 'joint_state',
      fields: ['position', 'velocity'],
      joint_ids: [0, 2],
    };
    const applied = applyEntityWrite(model, data, target, {
      joint_state__position: new Float32Array([0.25, -0.5]),
      joint_state__velocity: new Float32Array([1.5, -2.5]),
    });
    expect(applied).toBe(true);
    expect(data.qpos[0]).toBeCloseTo(0.25, 6);
    expect(data.qpos[1]).toBe(0); // untouched joint
    expect(data.qpos[2]).toBeCloseTo(-0.5, 6);
    expect(data.qvel[0]).toBeCloseTo(1.5, 6);
    expect(data.qvel[2]).toBeCloseTo(-2.5, 6);
  });

  it('"all" targets every joint', () => {
    const model = fakeModel(2, false);
    const data = fakeData(model);
    applyEntityWrite(
      model,
      data,
      { kind: 'joint_state', fields: ['position', 'velocity'], joint_ids: 'all' },
      { joint_state__position: new Float32Array([0.1, 0.2]) },
    );
    expect(data.qpos[0]).toBeCloseTo(0.1, 6);
    expect(data.qpos[1]).toBeCloseTo(0.2, 6);
  });

  it('sets (does not accumulate) — the graph already computed the final value', () => {
    const model = fakeModel(1, false);
    const data = fakeData(model);
    data.qpos[0] = 99;
    applyEntityWrite(
      model,
      data,
      { kind: 'joint_state', fields: ['position'], joint_ids: [0] },
      { joint_state__position: new Float32Array([0.5]) },
    );
    expect(data.qpos[0]).toBeCloseTo(0.5, 6);
  });
});

describe('applyEntityWrite: root pose / velocity', () => {
  it('writes the 7-vector pose into the free joint qpos', () => {
    const model = fakeModel(1);
    const data = fakeData(model);
    const pose = new Float32Array([1, 2, 3, 0, 0, 0, 1]);
    expect(
      applyEntityWrite(model, data, { kind: 'root_pose', fields: ['pose'] }, {
        root_pose__pose: pose,
      }),
    ).toBe(true);
    for (let i = 0; i < 7; i++) expect(data.qpos[i]).toBeCloseTo(pose[i], 6);
  });

  it('writes the 6-vector velocity into the free joint qvel', () => {
    const model = fakeModel(1);
    const data = fakeData(model);
    const vel = new Float32Array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]);
    expect(
      applyEntityWrite(model, data, { kind: 'root_velocity', fields: ['velocity'] }, {
        root_velocity__velocity: vel,
      }),
    ).toBe(true);
    for (let i = 0; i < 6; i++) expect(data.qvel[i]).toBeCloseTo(vel[i], 6);
  });

  it('degrades (returns false) on a fixed-base model instead of throwing', () => {
    const model = fakeModel(2, false);
    const data = fakeData(model);
    expect(
      applyEntityWrite(model, data, { kind: 'root_pose', fields: ['pose'] }, {
        root_pose__pose: new Float32Array(7),
      }),
    ).toBe(false);
  });

  it('ignores a short/absent value rather than writing garbage', () => {
    const model = fakeModel(1);
    const data = fakeData(model);
    expect(
      applyEntityWrite(model, data, { kind: 'root_pose', fields: ['pose'] }, {
        root_pose__pose: new Float32Array([1, 2, 3]),
      }),
    ).toBe(false);
    expect(applyEntityWrite(model, data, { kind: 'root_pose', fields: ['pose'] }, {})).toBe(
      false,
    );
  });
});

describe('applyEntityWrites', () => {
  it('applies both writes a LiftingCommand-shaped term emits', () => {
    const model = fakeModel(1);
    const data = fakeData(model);
    const targets: WriteTarget[] = [
      { kind: 'root_pose', entity: 'cube', fields: ['pose'] },
      { kind: 'root_velocity', entity: 'cube', fields: ['velocity'] },
    ];
    const applied = applyEntityWrites(model, data, targets, {
      root_pose__pose: new Float32Array([0.4, 0, 0.3, 1, 0, 0, 0]),
      root_velocity__velocity: new Float32Array(6),
    });
    expect(applied).toBe(2);
    expect(data.qpos[0]).toBeCloseTo(0.4, 6);
    expect(data.qpos[3]).toBeCloseTo(1, 6);
  });
});
