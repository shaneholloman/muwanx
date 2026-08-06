/**
 * Slot-reader parity against mjlab itself.
 *
 * `slotReader.test.ts` checks the reader's logic on a hand-built model where every
 * expected number is computed by hand. This file checks the thing that test
 * cannot: that the semantics are *mjlab's*. The fixture is dumped from two live,
 * stepped mjlab tasks (`tests/dump_slot_fixture.py`) and carries both the raw
 * `mjModel`/`mjData` arrays the reader indexes and mjlab's own
 * `env.scene[entity].data.<field>` value for each one.
 *
 * This closes the last gap in the ADR 0005 verification chain: the Python parity
 * harness proves each traced graph reproduces mjlab's term, and this proves the
 * browser feeds that graph the same numbers mjlab would.
 *
 * Regenerate the fixture whenever a field is added to the reader — the coverage
 * assertion at the bottom fails if a task's fields go unchecked.
 */
import { describe, expect, it } from 'vitest';

import fixture from './fixtures/slotFields.json';
import { createSlotReader, isReadableEntityField, type SlotReaderContext } from '../slotReader';

type EntityFixture = {
  fields: Record<string, number[]>;
  /** mjlab's per-episode randomized encoder bias, by unprefixed joint name. */
  encoder_bias: Record<string, number>;
};

type TaskFixture = {
  model: Record<string, number | number[]>;
  data: Record<string, number[]>;
  entities: Record<string, EntityFixture>;
  sensors: Record<string, number[]>;
};

const TASKS = fixture as unknown as Record<string, TaskFixture>;

function contextFor(task: TaskFixture): SlotReaderContext {
  const { names, ...rest } = task.model as Record<string, number | number[]>;
  const mjModel = {
    ...rest,
    names: Uint8Array.from(names as number[]).buffer,
  };
  const mjData: Record<string, Float64Array> = {};
  for (const [key, values] of Object.entries(task.data)) {
    mjData[key] = Float64Array.from(values);
  }
  return { mjModel, mjData } as unknown as SlotReaderContext;
}

/** mjData is float64 and the read casts to float32, so compare at float32 resolution. */
function expectClose(actual: Float32Array, expected: number[], label: string): void {
  expect(actual.length, `${label}: length`).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const tolerance = Math.max(1e-5, Math.abs(expected[i]) * 1e-5);
    expect(Math.abs(actual[i] - expected[i]), `${label}[${i}]`).toBeLessThan(tolerance);
  }
}

describe.each(Object.keys(TASKS))('slot reader vs mjlab — %s', taskId => {
  const task = TASKS[taskId];
  // The walking tasks randomize the `encoder_bias` that `joint_pos_biased` observes, so
  // the reader gets the same lookup. Merging every entity's is safe: joint names are unique.
  const jointBias = new Map<string, number>();
  for (const entity of Object.values(task.entities)) {
    for (const [joint, bias] of Object.entries(entity.encoder_bias)) {
      jointBias.set(joint, bias);
    }
  }
  const read = createSlotReader(() => contextFor(task), {
    jointBias: name => jointBias.get(name) ?? 0,
  });

  // mjlab attaches terrain with `prefix=""`, so it is not name-resolvable — asserted below.
  const cases: Array<[string, string, number[]]> = [];
  for (const [entity, { fields }] of Object.entries(task.entities)) {
    if (entity === 'terrain') continue;
    for (const [field, expected] of Object.entries(fields)) {
      if (isReadableEntityField(field)) cases.push([entity, field, expected]);
    }
  }

  it.each(cases)('%s.%s matches mjlab', (entity, field, expected) => {
    const value = read({ entity, field, input: `${entity}__${field}` });
    expect(value, `${entity}.${field} unreadable`).not.toBeNull();
    expectClose(value!, expected, `${entity}.${field}`);
  });

  const sensorCases = Object.entries(task.sensors);
  it.each(sensorCases)('sensor %s matches mjlab', (sensor, expected) => {
    const value = read({ sensor, input: `sensor__${sensor}` });
    expect(value, `sensor ${sensor} unreadable`).not.toBeNull();
    expectClose(value!, expected, `sensor ${sensor}`);
  });

  it('reports the prefix-free terrain as unavailable, not as another entity', () => {
    // `terrain` matching no prefix must not fall back and hand over the robot's joints.
    expect(read({ entity: 'terrain', field: 'joint_pos', input: 'terrain__joint_pos' })).toEqual(
      new Float32Array(0),
    );
    expect(read({ entity: 'terrain', field: 'root_link_pos_w', input: 'x' })).toBeNull();
  });

  it('covers every field the reader implements', () => {
    // Guards a field added to the reader but never compared: regenerate, and it appears.
    const checked = new Set(cases.map(([, field]) => field));
    for (const field of [
      'joint_pos',
      'joint_pos_biased',
      'joint_vel',
      'root_link_pos_w',
      'root_link_quat_w',
      'root_link_pose_w',
      'root_link_vel_w',
      'root_link_lin_vel_w',
      'root_link_ang_vel_w',
      'root_link_lin_vel_b',
      'root_link_ang_vel_b',
      'projected_gravity_b',
      'heading_w',
      'site_pos_w',
    ]) {
      expect(isReadableEntityField(field), `${field} not implemented`).toBe(true);
      expect(checked.has(field), `${field} not compared with mjlab`).toBe(true);
    }
  });
});
