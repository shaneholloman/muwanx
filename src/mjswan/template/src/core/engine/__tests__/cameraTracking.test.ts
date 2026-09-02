/** How a tracked body's motion reaches the desktop camera and the XR rig differently. */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { updateCameraFromData, type ViewerState } from '../viewer_config';

type MjData = import('mujoco').MjData;

/** Enough of `mjData` for the tracker: one body's world position, in MuJoCo's frame. */
function at(x: number, y: number, z: number): MjData {
  return { xpos: new Float64Array([0, 0, 0, x, y, z]) } as unknown as MjData;
}

function controlsAt(target: THREE.Vector3): OrbitControls {
  return { target } as unknown as OrbitControls;
}

function tracking(): {
  camera: THREE.PerspectiveCamera;
  rig: THREE.Group;
  controls: OrbitControls;
  state: ViewerState;
} {
  return {
    camera: new THREE.PerspectiveCamera(),
    rig: new THREE.Group(),
    controls: controlsAt(new THREE.Vector3()),
    state: { trackBodyId: 1, prevBodyPos: null },
  };
}

describe('updateCameraFromData without a rig', () => {
  it('gives the camera the body’s whole delta, vertical included', () => {
    const { camera, controls, state } = tracking();
    updateCameraFromData(at(0, 0, 0), camera, controls, state);
    updateCameraFromData(at(1, 2, 0.5), camera, controls, state);

    // MuJoCo (x, y, z) is three.js (x, z, -y).
    expect(camera.position.x).toBeCloseTo(1, 6);
    expect(camera.position.y).toBeCloseTo(0.5, 6);
    expect(camera.position.z).toBeCloseTo(-2, 6);
  });

  /** A chase view is meant to keep up with a reset, so this delta is not filtered. */
  it('follows a reset teleport', () => {
    const { camera, controls, state } = tracking();
    updateCameraFromData(at(0, 0, 0), camera, controls, state);
    updateCameraFromData(at(40, 0, 0), camera, controls, state);

    expect(camera.position.x).toBeCloseTo(40, 6);
  });
});

describe('updateCameraFromData with a rig', () => {
  it('moves the rig horizontally and leaves its height alone', () => {
    const { camera, rig, controls, state } = tracking();
    updateCameraFromData(at(0, 0, 0), camera, controls, state, rig);
    updateCameraFromData(at(0.2, 0.1, 0.4), camera, controls, state, rig);

    expect(rig.position.x).toBeCloseTo(0.2, 6);
    expect(rig.position.z).toBeCloseTo(-0.1, 6);
    // The vertical belongs to the ground under the viewer, not to the body.
    expect(rig.position.y).toBe(0);
    expect(camera.position.length()).toBe(0);
  });

  /** An episode reset draws a new spawn patch; carrying a viewer there is not tracking. */
  it('refuses a delta the size of a reset teleport', () => {
    const { camera, rig, controls, state } = tracking();
    updateCameraFromData(at(0, 0, 0), camera, controls, state, rig);
    updateCameraFromData(at(40, 12, 0), camera, controls, state, rig);

    expect(rig.position.length()).toBe(0);
  });

  it('picks tracking back up from the new spawn', () => {
    const { camera, rig, controls, state } = tracking();
    updateCameraFromData(at(0, 0, 0), camera, controls, state, rig);
    updateCameraFromData(at(40, 0, 0), camera, controls, state, rig);
    updateCameraFromData(at(40.3, 0, 0), camera, controls, state, rig);

    expect(rig.position.x).toBeCloseTo(0.3, 6);
  });

  /** The desktop view has to be coherent again the moment the session ends. */
  it('keeps the orbit target on the body through both', () => {
    const { camera, rig, controls, state } = tracking();
    updateCameraFromData(at(0, 0, 0), camera, controls, state, rig);
    updateCameraFromData(at(40, 0, 2), camera, controls, state, rig);

    expect(controls.target.x).toBeCloseTo(40, 6);
    expect(controls.target.y).toBeCloseTo(2, 6);
  });
});

describe('updateCameraFromData with nothing tracked', () => {
  it('leaves every viewpoint where it was', () => {
    const { camera, rig, controls } = tracking();
    const state: ViewerState = { trackBodyId: null, prevBodyPos: null };
    updateCameraFromData(at(0, 0, 0), camera, controls, state, rig);
    updateCameraFromData(at(5, 5, 5), camera, controls, state, rig);

    expect(rig.position.length()).toBe(0);
    expect(camera.position.length()).toBe(0);
    expect(controls.target.length()).toBe(0);
  });
});
