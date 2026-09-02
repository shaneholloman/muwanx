/** The half of XR locomotion that runs without a headset: what a stick reading does to
 * the rig. The rig stands in for a session's reference space, the camera for the head. */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { XrLocomotion } from '../locomotion';

/** Enough of an `XRSession` for `update`: the input sources it reads gamepads off. */
function session(sources: Array<{ handedness: string; axes: number[] }>): XRSession {
  return {
    inputSources: sources.map((source) => ({
      handedness: source.handedness,
      gamepad: { axes: source.axes },
    })),
  } as unknown as XRSession;
}

const left = (x: number, y: number) => session([{ handedness: 'left', axes: [0, 0, x, y] }]);
const right = (x: number) => session([{ handedness: 'right', axes: [0, 0, x, 0] }]);

/** The module's own speeds, in m/s and deg/s, and a frame short enough to dodge its clamp. */
const MOVE_SPEED = 1.5;
const TURN_SPEED = 90;
const FRAME = 0.05;
const STEP = MOVE_SPEED * FRAME;
/** What a full-deflection stick turns through in one `drive`, in radians. */
const SWEEP = ((TURN_SPEED * FRAME) / 180) * Math.PI;

/** A head 1.6 m up and off the play-area centre, facing -Z as a fresh camera does. */
function rigged(): { rig: THREE.Group; camera: THREE.PerspectiveCamera; move: XrLocomotion } {
  const rig = new THREE.Group();
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(0.4, 1.6, -0.3);
  rig.add(camera);
  rig.updateMatrixWorld(true);
  return { rig, camera, move: new XrLocomotion(rig, camera) };
}

/** One frame's worth of input. The frame time comes from the caller's clock. */
function drive(move: XrLocomotion, input: XRSession, seconds = FRAME): void {
  move.update(input, seconds);
}

describe('XrLocomotion sliding', () => {
  it('slides along the head’s heading, at the stick’s scale', () => {
    const { rig, move } = rigged();
    drive(move, left(0, -1));

    // Stick forward is -1 and the camera faces -Z.
    expect(rig.position.x).toBeCloseTo(0, 6);
    expect(rig.position.y).toBeCloseTo(0, 6);
    expect(rig.position.z).toBeCloseTo(-STEP, 6);

    const half = rigged();
    drive(half.move, left(0, -0.5));
    expect(half.rig.position.z).toBeCloseTo(-STEP / 2, 6);
  });

  it('turns with the head rather than the rig', () => {
    const { rig, camera, move } = rigged();
    camera.rotateY(-Math.PI / 2); // face +X
    rig.updateMatrixWorld(true);
    drive(move, left(0, -1));

    expect(rig.position.x).toBeCloseTo(STEP, 6);
    expect(rig.position.z).toBeCloseTo(0, 6);
  });

  it('stays level, and stays put with no heading to slide along', () => {
    const { rig, camera, move } = rigged();
    camera.rotateX(-Math.PI / 2); // look straight down
    rig.updateMatrixWorld(true);
    drive(move, left(0, -1));

    expect(rig.position.length()).toBeCloseTo(0, 6);
    expect(Number.isNaN(rig.position.x)).toBe(false);
  });

  it('ignores a resting stick, and a frame with no time in it', () => {
    const { rig, move } = rigged();
    drive(move, left(0.1, -0.1));
    expect(rig.position.length()).toBeCloseTo(0, 6);

    const first = rigged();
    first.move.update(left(0, -1), 0);
    expect(first.rig.position.length()).toBeCloseTo(0, 6);
  });

  it('does nothing without a session', () => {
    const { rig, move } = rigged();
    move.update(null, FRAME);
    expect(rig.position.length()).toBeCloseTo(0, 6);
  });
});

describe('XrLocomotion turning', () => {
  /** The reason turns pivot on the head: the viewer must not be swung through the scene. */
  it('leaves the head where it stands', () => {
    const { rig, camera, move } = rigged();
    rig.position.set(2, 0, 3);
    rig.updateMatrixWorld(true);
    const before = camera.getWorldPosition(new THREE.Vector3());

    drive(move, right(1));
    rig.updateMatrixWorld(true);
    const after = camera.getWorldPosition(new THREE.Vector3());

    expect(after.distanceTo(before)).toBeCloseTo(0, 6);
    // Holding the head still took a shift of the rig, not just a rotation of it.
    expect(rig.position.distanceTo(new THREE.Vector3(2, 0, 3))).toBeGreaterThan(0.001);
  });

  it('turns the view right for a stick pushed right, and left for left', () => {
    const { camera, rig, move } = rigged();
    drive(move, right(1));
    rig.updateMatrixWorld(true);

    const heading = camera.getWorldDirection(new THREE.Vector3());
    // Right of -Z is -Z swung toward +X.
    expect(heading.x).toBeCloseTo(Math.sin(SWEEP), 6);
    expect(heading.z).toBeCloseTo(-Math.cos(SWEEP), 6);

    const other = rigged();
    drive(other.move, right(-1));
    other.rig.updateMatrixWorld(true);
    expect(other.camera.getWorldDirection(new THREE.Vector3()).x).toBeCloseTo(-Math.sin(SWEEP), 6);
  });

  /** The point of holding rather than snapping: it does not stop after one step. */
  it('keeps turning for as long as the stick is held', () => {
    const { rig, move } = rigged();
    const held = right(1);

    const angles: number[] = [];
    for (let frame = 0; frame < 4; frame++) {
      move.update(held, FRAME);
      angles.push(rig.quaternion.angleTo(new THREE.Quaternion()));
    }

    for (const [i, angle] of angles.entries()) {
      expect(angle).toBeCloseTo(SWEEP * (i + 1), 6);
    }
  });

  it('turns at the stick’s own rate, and stops when it is let go', () => {
    const { rig, move } = rigged();
    drive(move, right(0.5));
    expect(rig.quaternion.angleTo(new THREE.Quaternion())).toBeCloseTo(SWEEP / 2, 6);

    const held = rig.quaternion.clone();
    move.update(right(0), FRAME);
    expect(rig.quaternion.angleTo(held)).toBeCloseTo(0, 6);
  });

  it('ignores a resting stick', () => {
    const { rig, move } = rigged();
    drive(move, right(0.1));
    expect(rig.quaternion.angleTo(new THREE.Quaternion())).toBeCloseTo(0, 6);
  });
});

describe('XrLocomotion reset', () => {
  it('returns the rig to the origin when a session ends', () => {
    const { rig, move } = rigged();
    drive(move, left(1, -1));
    drive(move, right(1));
    rig.updateMatrixWorld(true);

    move.reset();

    expect(rig.position.length()).toBeCloseTo(0, 6);
    expect(rig.quaternion.angleTo(new THREE.Quaternion())).toBeCloseTo(0, 6);
  });

});
