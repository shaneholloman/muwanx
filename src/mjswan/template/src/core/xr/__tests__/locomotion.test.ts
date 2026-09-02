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

/** The module's own speed, in m/s, and a frame short enough to dodge its clamp. */
const MOVE_SPEED = 1.5;
const FRAME = 0.05;
const STEP = MOVE_SPEED * FRAME;

/** A head 1.6 m up and off the play-area centre, facing -Z as a fresh camera does. */
function rigged(): { rig: THREE.Group; camera: THREE.PerspectiveCamera; move: XrLocomotion } {
  const rig = new THREE.Group();
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(0.4, 1.6, -0.3);
  rig.add(camera);
  rig.updateMatrixWorld(true);
  return { rig, camera, move: new XrLocomotion(rig, camera) };
}

/** Two frames: the first only starts the clock, the second is the step being measured. */
function drive(move: XrLocomotion, input: XRSession, seconds = FRAME): void {
  move.update(input, 0);
  move.update(input, seconds * 1000);
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

  it('ignores a resting stick, and the frame that only starts the clock', () => {
    const { rig, move } = rigged();
    drive(move, left(0.1, -0.1));
    expect(rig.position.length()).toBeCloseTo(0, 6);

    const first = rigged();
    first.move.update(left(0, -1), 0);
    expect(first.rig.position.length()).toBeCloseTo(0, 6);
  });

  it('clamps a frame the browser slept through', () => {
    const { rig, move } = rigged();
    drive(move, left(0, -1), 30);
    // 0.1 s of the 30 s gap, not the whole tab-away.
    expect(rig.position.z).toBeCloseTo(-MOVE_SPEED * 0.1, 6);
  });
});

describe('XrLocomotion snap turns', () => {
  /** The reason turns pivot on the head: the viewer must not be swung through the scene. */
  it('leaves the head where it stands', () => {
    const { rig, camera, move } = rigged();
    rig.position.set(2, 0, 3);
    rig.updateMatrixWorld(true);
    const before = camera.getWorldPosition(new THREE.Vector3());

    move.update(right(1));
    rig.updateMatrixWorld(true);
    const after = camera.getWorldPosition(new THREE.Vector3());

    expect(after.distanceTo(before)).toBeCloseTo(0, 6);
    // Holding the head still took a shift of the rig, not just a rotation of it.
    expect(rig.position.distanceTo(new THREE.Vector3(2, 0, 3))).toBeGreaterThan(0.01);
  });

  it('turns the view right for a stick pushed right', () => {
    const { rig, camera, move } = rigged();
    move.update(right(1));
    rig.updateMatrixWorld(true);

    const heading = camera.getWorldDirection(new THREE.Vector3());
    // 30° right of -Z is -Z rotated toward +X.
    expect(heading.x).toBeCloseTo(Math.sin(Math.PI / 6), 6);
    expect(heading.z).toBeCloseTo(-Math.cos(Math.PI / 6), 6);

    const other = rigged();
    other.move.update(right(-1));
    other.rig.updateMatrixWorld(true);
    expect(other.camera.getWorldDirection(new THREE.Vector3()).x).toBeCloseTo(
      -Math.sin(Math.PI / 6),
      6
    );
  });

  it('fires once per push, and re-arms when the stick comes back', () => {
    const { rig, move } = rigged();
    for (let frame = 0; frame < 10; frame++) {
      move.update(right(1));
    }
    const oneTurn = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 1, 0),
      -Math.PI / 6
    );
    expect(rig.quaternion.angleTo(oneTurn)).toBeCloseTo(0, 6);

    move.update(right(0));
    move.update(right(1));
    expect(rig.quaternion.angleTo(oneTurn)).toBeGreaterThan(0.1);
  });

  it('holds still for a stick short of the threshold', () => {
    const { rig, move } = rigged();
    move.update(right(0.5));
    expect(rig.quaternion.angleTo(new THREE.Quaternion())).toBeCloseTo(0, 6);
  });
});

describe('XrLocomotion reset', () => {
  it('returns the rig to the origin when a session ends', () => {
    const { rig, move } = rigged();
    drive(move, left(1, -1));
    move.update(right(1));
    rig.updateMatrixWorld(true);

    move.reset();

    expect(rig.position.length()).toBeCloseTo(0, 6);
    expect(rig.quaternion.angleTo(new THREE.Quaternion())).toBeCloseTo(0, 6);
  });

  it('drops the clock with the session, so the next one starts from a standstill', () => {
    const { rig, move } = rigged();
    move.update(left(0, -1), 0);
    move.update(null);
    move.update(left(0, -1), 10_000);

    expect(rig.position.length()).toBeCloseTo(0, 6);
  });
});
