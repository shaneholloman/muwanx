import { describe, expect, it } from 'vitest';
import { ContactSensorSet, type ContactSensorDescriptor } from '../contact';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** `ee_ground_collision` as the build emits it: one primary, one slot, 4 deep. */
const DESCRIPTOR: ContactSensorDescriptor = {
  kind: 'contact',
  num_slots: 1,
  history_length: 4,
  history_fields: ['force'],
  fields: {
    found: { sensors: ['ee_ground_collision_link_6_found'], dim: 1 },
    force: { sensors: ['ee_ground_collision_link_6_force'], dim: 3 },
  },
};

const WINDOWS: Record<string, { adr: number; dim: number }> = {
  ee_ground_collision_link_6_found: { adr: 0, dim: 1 },
  ee_ground_collision_link_6_force: { adr: 1, dim: 3 },
};

function fixture() {
  const mjData = { sensordata: new Float64Array(4) } as unknown as MjData;
  const mjModel = {} as unknown as MjModel;
  const lookup = (sensor: string) => WINDOWS[sensor] ?? null;
  const set = new ContactSensorSet({ ee_ground_collision: DESCRIPTOR });
  const push = (force: [number, number, number]) => {
    mjData.sensordata[1] = force[0];
    mjData.sensordata[2] = force[1];
    mjData.sensordata[3] = force[2];
    set.advance(mjModel, mjData, lookup);
  };
  const read = (field: string) => set.read('ee_ground_collision', field, mjModel, mjData, lookup);
  return { set, mjData, push, read };
}

describe('ContactSensorSet', () => {
  it('reads a field straight out of its sensordata window', () => {
    const { mjData, read } = fixture();
    mjData.sensordata[0] = 2;
    mjData.sensordata[1] = 11;
    mjData.sensordata[3] = 13;

    expect(Array.from(read('found')!)).toEqual([2]);
    expect(Array.from(read('force')!)).toEqual([11, 0, 13]);
  });

  it('keeps the newest reading at index 0, oldest last', () => {
    // mjlab's `_update_history`: roll by one along the history axis, write index 0.
    const { push, read } = fixture();
    push([1, 0, 0]);
    push([2, 0, 0]);
    push([3, 0, 0]);

    const history = Array.from(read('force_history')!);
    expect(history.filter((_, i) => i % 3 === 0)).toEqual([3, 2, 1, 0]);
  });

  it('drops the reading that falls off the end', () => {
    const { push, read } = fixture();
    for (const x of [1, 2, 3, 4, 5]) push([x, 0, 0]);

    expect(Array.from(read('force_history')!).filter((_, i) => i % 3 === 0)).toEqual([
      5, 4, 3, 2,
    ]);
  });

  it('forgets the past on reset, so a pre-reset force cannot re-fire a term', () => {
    const { set, push, read } = fixture();
    push([9, 9, 9]);

    set.reset();

    expect(Array.from(read('force_history')!)).toEqual(new Array(12).fill(0));
  });

  it('has nothing to say about a field with no buffer', () => {
    const { push, read } = fixture();
    push([1, 2, 3]);
    expect(read('found_history')).toBeNull();
  });

  it('ignores a sensor it does not own, leaving the caller to warn', () => {
    const { set, mjData } = fixture();
    const mjModel = {} as unknown as MjModel;
    expect(set.read('height_scan', 'distances', mjModel, mjData, () => null)).toBeNull();
  });
});
