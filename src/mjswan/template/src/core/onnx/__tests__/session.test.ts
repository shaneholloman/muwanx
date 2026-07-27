/**
 * Input-slot naming contract (ADR 0005).
 *
 * The build decides each slot's graph input name and ships it in the slot's own
 * `input` field (`mjswan.compile.tracer.slot_to_json`), so the runtime never has
 * to reproduce the naming scheme — which it could not do for sensor slots, whose
 * MJCF-path names get folded to identifiers at build time.
 */
import { describe, expect, it } from 'vitest';

import { slotInputName } from '../session';

describe('slotInputName', () => {
  it('uses the build-supplied input name for an entity-data slot', () => {
    expect(
      slotInputName({ entity: 'robot', field: 'heading_w', input: 'robot__heading_w' }),
    ).toBe('robot__heading_w');
  });

  it('uses the build-supplied input name for a sensor slot', () => {
    // The sensor's real name keeps its MJCF path; only the graph input name is folded.
    expect(
      slotInputName({ sensor: 'robot/imu_lin_vel', input: 'sensor__robot_imu_lin_vel' }),
    ).toBe('sensor__robot_imu_lin_vel');
  });

  it('uses the build-supplied input name for a command-state slot', () => {
    expect(
      slotInputName({
        command: 'lift_height',
        field: 'target_pos',
        input: 'command__lift_height_target_pos',
      }),
    ).toBe('command__lift_height_target_pos');
  });

  it('falls back to the legacy entity__field scheme when input is absent', () => {
    expect(slotInputName({ entity: 'robot', field: 'joint_pos' })).toBe('robot__joint_pos');
  });

  it('falls back to the entity placeholder for a null entity', () => {
    expect(slotInputName({ entity: null, field: 'joint_pos' })).toBe('entity__joint_pos');
  });
});
