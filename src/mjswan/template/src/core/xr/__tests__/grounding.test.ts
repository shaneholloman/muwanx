/** Where the rig's height comes from once the ground under the viewer decides it. */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { updateRigGrounding } from '../grounding';

const CLIMB_SPEED = 3;
const FRAME = 0.05;
/** The module's own margin: the head is never allowed nearer the ground than this. */
const MIN_HEAD_CLEARANCE = 0.15;

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;

/**
 * A `mj_ray` that answers from a height function of MuJoCo x/y, so a test can shape the
 * ground under the viewer. `height` returning null is a ray that hit nothing.
 */
function mujocoWith(height: (x: number, y: number) => number | null): MainModule {
  return {
    mj_ray: (
      _model: unknown,
      _data: unknown,
      origin: number[],
      direction: number[],
      _group: number[],
      _staticFlag: number,
      _exclude: number,
      _geomId: Int32Array
    ) => {
      expect(direction).toEqual([0, 0, -1]);
      const surface = height(origin[0], origin[1]);
      if (surface === null) {
        return -1;
      }
      expect(origin[2]).toBeGreaterThan(surface); // or the ray starts inside the ground
      return origin[2] - surface;
    },
  } as unknown as MainModule;
}

const MODEL = {} as MjModel;
const DATA = {} as MjData;

/** A rig with the head 1.5 m up: a headset floor calibration that is not 1.7. */
function rigged(headHeight = 1.5): {
  rig: THREE.Group;
  camera: THREE.PerspectiveCamera;
  /** One frame of grounding, bound to this rig. */
  ground: (mujoco: MainModule, m?: MjModel | null, d?: MjData | null) => void;
  /** The head's world height, which is what the module is really placing. */
  headY: () => number;
} {
  const rig = new THREE.Group();
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(0, headHeight, 0);
  rig.add(camera);
  rig.updateMatrixWorld(true);
  return {
    rig,
    camera,
    ground: (mujoco, m = MODEL, d = DATA) =>
      updateRigGrounding(rig, camera, mujoco, m, d, FRAME),
    headY: () => {
      rig.updateMatrixWorld(true);
      return camera.getWorldPosition(new THREE.Vector3()).y;
    },
  };
}

/** Run frames until the height settles, or fail loudly rather than loop forever. */
function settle(
  ground: (mujoco: MainModule) => void,
  mujoco: MainModule,
  rig: THREE.Group,
  frames = 200
): void {
  for (let i = 0; i < frames; i++) {
    const before = rig.position.y;
    ground(mujoco);
    rig.updateMatrixWorld(true);
    if (Math.abs(rig.position.y - before) < 1e-9) {
      return;
    }
  }
  throw new Error('the rig height never settled');
}

describe('updateRigGrounding', () => {
  /** The viewer's height in the scene is their own, so only the floor is placed. */
  it('puts the floor on the ground and leaves the headset its own height', () => {
    for (const headHeight of [1.5, 1.7, 1.9]) {
      const { rig, ground, headY } = rigged(headHeight);
      settle(ground, mujocoWith(() => 0), rig);
      expect(rig.position.y).toBeCloseTo(0, 6);
      expect(headY()).toBeCloseTo(headHeight, 6);
    }
  });

  it('measures from the ground, not from z = 0', () => {
    const { rig, ground, headY } = rigged(1.5);
    settle(ground, mujocoWith(() => 1.2), rig);
    expect(headY()).toBeCloseTo(1.2 + 1.5, 6);

    const pit = rigged(1.5);
    settle(pit.ground, mujocoWith(() => -0.6), pit.rig);
    expect(pit.headY()).toBeCloseTo(-0.6 + 1.5, 6);
  });

  /** Head movement is 1:1 because the module never corrects the eye height at all. */
  it('lets the viewer crouch instead of cancelling it', () => {
    const { rig, camera, ground, headY } = rigged();
    const mujoco = mujocoWith(() => 0);
    settle(ground, mujoco, rig);
    const standing = headY();

    camera.position.y -= 0.5;
    rig.updateMatrixWorld(true);
    ground(mujoco);

    expect(headY()).toBeCloseTo(standing - 0.5, 6);
  });

  /** A rise the head still clears, so the rate limit is what decides — not the clamp below. */
  it('limits how fast the floor may travel', () => {
    const { rig, ground } = rigged();
    settle(ground, mujocoWith(() => 0), rig);
    expect(rig.position.y).toBeCloseTo(0, 6);

    const step = mujocoWith(() => 0.5);
    ground(step);
    expect(rig.position.y).toBeCloseTo(CLIMB_SPEED * FRAME, 6);
    ground(step);
    expect(rig.position.y).toBeCloseTo(2 * CLIMB_SPEED * FRAME, 6);

    settle(ground, step, rig);
    expect(rig.position.y).toBeCloseTo(0.5, 6);
  });

  /** The rate limit must not leave the head inside the ground while it catches up. */
  it('lifts the head clear of the ground at once', () => {
    const { rig, ground, headY } = rigged();
    // Standing in a hollow, then the ground under the viewer jumps well above the head.
    ground(mujocoWith(() => 8));
    rig.updateMatrixWorld(true);

    expect(headY()).toBeCloseTo(8 + MIN_HEAD_CLEARANCE, 6);
  });

  it('keeps the height it had when the ray hits nothing', () => {
    const { rig, ground } = rigged();
    settle(ground, mujocoWith(() => 0.4), rig);
    const held = rig.position.y;

    ground(mujocoWith(() => null));
    expect(rig.position.y).toBeCloseTo(held, 6);
  });

  it('samples the ground under the head, not under the rig origin', () => {
    const { rig, camera, ground, headY } = rigged();
    camera.position.set(0.9, 1.5, -0.4);
    rig.position.set(4, 0, 7);
    rig.updateMatrixWorld(true);

    // three.js (x, y, z) is MuJoCo (x, -z, y): the head sits at MuJoCo (4.9, -6.6).
    settle(
      ground,
      mujocoWith((x, y) => {
        expect(x).toBeCloseTo(4.9, 6);
        expect(y).toBeCloseTo(-6.6, 6);
        return 2;
      }),
      rig
    );

    expect(headY()).toBeCloseTo(2 + 1.5, 6);
  });

  it('does nothing without a model', () => {
    const { rig, ground } = rigged();
    ground(mujocoWith(() => 5), null, DATA);
    ground(mujocoWith(() => 5), MODEL, null);
    expect(rig.position.y).toBe(0);
  });

  /** Nothing is remembered between frames, so a taller viewer needs no re-measuring. */
  it('follows a head height that changes under it', () => {
    const { rig, camera, ground, headY } = rigged(1.5);
    const mujoco = mujocoWith(() => 0);
    settle(ground, mujoco, rig);

    camera.position.y = 1.9;
    rig.updateMatrixWorld(true);
    settle(ground, mujoco, rig);

    expect(rig.position.y).toBeCloseTo(0, 6);
    expect(headY()).toBeCloseTo(1.9, 6);
  });
});
