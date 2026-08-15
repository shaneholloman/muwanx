/**
 * The debug-drawing evaluator, checked against the math it restates: mjlab applies the
 * frame transform in MuJoCo coordinates and converts to three's only after.
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { CommandDebugVisuals, type VizPrimitive } from '../debugViz';
import type { OnnxInputSlot } from '../../onnx/session';

const NO_STATE = () => null;

/** Robot yawed +90°, standing at (1, 2, 0), drifting forward at 1 m/s in its own frame. */
function robotSlots(): (slot: OnnxInputSlot) => Float32Array | null {
  const halfSqrt2 = Math.SQRT1_2;
  const fields: Record<string, number[]> = {
    root_link_pos_w: [1, 2, 0],
    root_link_quat_w: [halfSqrt2, 0, 0, halfSqrt2], // (w, x, y, z)
    root_link_lin_vel_b: [1, 0, 0],
  };
  return slot => {
    const value = fields[slot.field ?? ''];
    return value ? Float32Array.from(value) : null;
  };
}

describe('CommandDebugVisuals: spheres', () => {
  const SPHERE: VizPrimitive = {
    shape: 'sphere',
    radius: 0.03,
    color: [1, 0.5, 0, 0.3],
    origin: { state: 'target_pos' },
  };

  it('starts hidden and shows at the state field once enabled', () => {
    const scene = new THREE.Scene();
    const visuals = new CommandDebugVisuals('lift_height', [SPHERE], scene);
    expect(scene.children[0].visible).toBe(false);

    visuals.update(true, () => Float32Array.from([0.4, 0.1, 0.3]));
    const marker = scene.children[0];
    expect(marker.visible).toBe(true);
    // mjcToThreeCoordinate: (x, z, -y).
    expect(marker.position.x).toBeCloseTo(0.4, 6);
    expect(marker.position.y).toBeCloseTo(0.3, 6);
    expect(marker.position.z).toBeCloseTo(-0.1, 6);
  });

  it('hides when debug_vis is off', () => {
    const scene = new THREE.Scene();
    const visuals = new CommandDebugVisuals('lift_height', [SPHERE], scene);
    visuals.update(false, () => Float32Array.from([0.4, 0.1, 0.3]));
    expect(scene.children[0].visible).toBe(false);
  });

  it('hides rather than drawing at the origin when the source is missing', () => {
    // An uncomputed term would otherwise pin its marker at (0, 0, 0).
    const scene = new THREE.Scene();
    const visuals = new CommandDebugVisuals('lift_height', [SPHERE], scene);
    visuals.update(true, NO_STATE);
    expect(scene.children[0].visible).toBe(false);
  });
});

describe('CommandDebugVisuals: arrows', () => {
  const FRAME = {
    entity: 'robot',
    pos_field: 'root_link_pos_w',
    quat_field: 'root_link_quat_w',
  };
  const COMMAND_ARROW: VizPrimitive = {
    shape: 'arrow',
    color: [0.2, 0.2, 0.6, 0.6],
    width: 0.015,
    frame: FRAME,
    origin: { const: [0, 0, 0.1] },
    vector: { state: 'vel_command_b', components: [0, 1, null], scale: 0.5 },
  };

  function draw(primitive: VizPrimitive, command: number[]): THREE.Object3D {
    const scene = new THREE.Scene();
    const visuals = new CommandDebugVisuals('twist', [primitive], scene);
    visuals.update(true, () => Float32Array.from(command), robotSlots());
    return scene.children[0];
  }

  it('starts at the frame origin and points along the rotated command', () => {
    // Body +x is world +y at yaw 90°, so a forward command draws along world +y.
    const arrow = draw(COMMAND_ARROW, [2, 0, 0]);
    expect(arrow.visible).toBe(true);
    // Local (0, 0, 0.1) + robot at (1, 2, 0), in three coordinates (x, z, -y).
    expect(arrow.position.x).toBeCloseTo(1, 6);
    expect(arrow.position.y).toBeCloseTo(0.1, 6);
    expect(arrow.position.z).toBeCloseTo(-2, 6);

    const direction = new THREE.Vector3(0, 1, 0).applyQuaternion(arrow.quaternion);
    // World +y is three's -z.
    expect(direction.x).toBeCloseTo(0, 6);
    expect(direction.y).toBeCloseTo(0, 6);
    expect(direction.z).toBeCloseTo(-1, 6);
  });

  it('scales its length by the cfg scale, not by the raw command', () => {
    const [shaft, head] = draw(COMMAND_ARROW, [2, 0, 0]).children;
    // 2 m/s * 0.5 = 1 m, split 80/20 between shaft and head.
    expect(shaft.scale.y).toBeCloseTo(0.8, 6);
    expect(head.scale.y).toBeCloseTo(0.2, 6);
    expect(head.position.y).toBeCloseTo(0.8, 6);
  });

  it('takes only the components it declares', () => {
    // The angular arrow reads command[2] into z; a leaked xy would tilt it.
    const angular: VizPrimitive = {
      ...COMMAND_ARROW,
      vector: { state: 'vel_command_b', components: [null, null, 2], scale: 0.5 },
    };
    const direction = new THREE.Vector3(0, 1, 0).applyQuaternion(
      draw(angular, [9, 9, 1]).quaternion,
    );
    // Body +z is world +z is three's +y — unaffected by the yaw.
    expect(direction.y).toBeCloseTo(1, 6);
  });

  it('reads an entity field through the slot reader', () => {
    const actual: VizPrimitive = {
      ...COMMAND_ARROW,
      vector: { entity: 'robot', field: 'root_link_lin_vel_b', components: [0, 1, null], scale: 0.5 },
    };
    const [shaft] = draw(actual, []).children;
    expect(shaft.scale.y).toBeCloseTo(0.4, 6); // 1 m/s * 0.5, minus the head
  });

  it('hides a zero-length arrow rather than drawing a NaN direction', () => {
    expect(draw(COMMAND_ARROW, [0, 0, 0]).visible).toBe(false);
  });

  it('hides when the frame cannot be read', () => {
    const scene = new THREE.Scene();
    const visuals = new CommandDebugVisuals('twist', [COMMAND_ARROW], scene);
    visuals.update(true, () => Float32Array.from([1, 0, 0]), () => null);
    expect(scene.children[0].visible).toBe(false);
  });
});

describe('CommandDebugVisuals: lifecycle', () => {
  it('removes every primitive from the scene on dispose', () => {
    const scene = new THREE.Scene();
    const visuals = new CommandDebugVisuals(
      'twist',
      [
        { shape: 'sphere', radius: 0.03, color: [1, 0, 0, 1], origin: { const: [0, 0, 0] } },
        { shape: 'arrow', color: [0, 1, 0, 1], origin: { const: [0, 0, 0] }, vector: { const: [1, 0, 0] } },
      ],
      scene,
    );
    expect(scene.children.length).toBe(2);
    visuals.dispose();
    expect(scene.children.length).toBe(0);
  });
});
