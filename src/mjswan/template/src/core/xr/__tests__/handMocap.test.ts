/** The two halves of hand tracking that run without a headset: the injected MJCF, and
 * the frame swizzle the per-step writes go through. */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { DEFAULT_HAND_JOINTS, injectHandMocapXml } from '../handMocap';
import { threeToMjcQuaternion } from '../../scene/coordinate';
import { getQuaternion } from '../../scene/scene';

const MINIMAL =
  '<mujoco>\n  <worldbody>\n    <geom type="plane" size="5 5 .1"/>\n  </worldbody>\n</mujoco>';

/** Quaternions are double covers: `q` and `-q` are the same rotation. */
function expectSameRotation(a: readonly number[], b: readonly number[]): void {
  const dot = a.reduce((sum, value, i) => sum + value * b[i], 0);
  expect(Math.abs(dot)).toBeCloseTo(1, 6);
}

describe('injectHandMocapXml', () => {
  it('appends one mocap target and one welded fingertip per joint, per hand', () => {
    const xml = injectHandMocapXml(MINIMAL);
    const pairs = DEFAULT_HAND_JOINTS.length * 2;

    expect(xml.match(/mocap="true"/g)).toHaveLength(pairs);
    expect(xml.match(/<freejoint\/>/g)).toHaveLength(pairs);
    expect(xml.match(/<weld /g)).toHaveLength(pairs);
    for (const joint of DEFAULT_HAND_JOINTS) {
      expect(xml).toContain(`name="mjswan_xr0_${joint}_target"`);
      expect(xml).toContain(`name="mjswan_xr1_${joint}_tip"`);
    }
  });

  // A mocap target adds no `qpos`, and the block is appended, so neither can move the
  // robot's own free joint off `qpos[0]`, where `PolicyStateBuilder` reads it.
  it('leaves the original model ahead of the block, and one closing tag', () => {
    const xml = injectHandMocapXml(MINIMAL);
    expect(xml.indexOf('type="plane"')).toBeLessThan(xml.indexOf('mocap="true"'));
    expect(xml.match(/<\/mujoco>/g)).toHaveLength(1);
    expect(xml.endsWith('</mujoco>')).toBe(true);
  });

  it('refuses XML it cannot close', () => {
    expect(() => injectHandMocapXml('<mujoco>')).toThrow(/closing/);
  });
});

describe('threeToMjcQuaternion', () => {
  it('inverts the renderer swizzle', () => {
    const mjcQuats = [
      [1, 0, 0, 0],
      [Math.SQRT1_2, 0, 0, Math.SQRT1_2],
      [0.5, 0.5, 0.5, 0.5],
      [0.2, -0.4, 0.8, 0.4],
    ];
    for (const wxyz of mjcQuats) {
      const norm = Math.hypot(...wxyz);
      const mjc = wxyz.map((v) => v / norm);
      const three = getQuaternion(new Float32Array(mjc), 0, new THREE.Quaternion());
      expectSameRotation(threeToMjcQuaternion(three), mjc);
    }
  });

  it('maps a three.js yaw onto a MuJoCo yaw', () => {
    // MuJoCo is z-up and three.js y-up, so +90° about y is +90° about z.
    const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
    expectSameRotation(threeToMjcQuaternion(yaw), [Math.SQRT1_2, 0, 0, Math.SQRT1_2]);
  });
});
