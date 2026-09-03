/** How a tracked body's motion reaches the desktop camera, and stops at an XR session. */
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
  controls: OrbitControls;
  state: ViewerState;
} {
  return {
    camera: new THREE.PerspectiveCamera(),
    controls: controlsAt(new THREE.Vector3()),
    state: { trackBodyId: 1, prevBodyPos: null },
  };
}

describe('updateCameraFromData on the desktop', () => {
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

describe('updateCameraFromData while a session is presenting', () => {
  it('leaves the viewpoint alone, however far the body walks', () => {
    const { camera, controls, state } = tracking();
    updateCameraFromData(at(0, 0, 0), camera, controls, state, true);
    updateCameraFromData(at(0.2, 0.1, 0.4), camera, controls, state, true);
    updateCameraFromData(at(3, 2, 1), camera, controls, state, true);

    expect(camera.position.length()).toBe(0);
  });

  /** The desktop view has to point at the body again the moment the session ends. */
  it('keeps the orbit target on the body', () => {
    const { camera, controls, state } = tracking();
    updateCameraFromData(at(0, 0, 0), camera, controls, state, true);
    updateCameraFromData(at(40, 0, 2), camera, controls, state, true);

    expect(controls.target.x).toBeCloseTo(40, 6);
    expect(controls.target.y).toBeCloseTo(2, 6);
  });

  it('hands the camera back to tracking when the session ends', () => {
    const { camera, controls, state } = tracking();
    updateCameraFromData(at(0, 0, 0), camera, controls, state, true);
    updateCameraFromData(at(1, 0, 0), camera, controls, state, true);
    updateCameraFromData(at(1.5, 0, 0), camera, controls, state);

    expect(camera.position.x).toBeCloseTo(0.5, 6);
  });
});

describe('updateCameraFromData with nothing tracked', () => {
  it('leaves every viewpoint where it was', () => {
    const { camera, controls } = tracking();
    const state: ViewerState = { trackBodyId: null, prevBodyPos: null };
    updateCameraFromData(at(0, 0, 0), camera, controls, state);
    updateCameraFromData(at(5, 5, 5), camera, controls, state);

    expect(camera.position.length()).toBe(0);
    expect(controls.target.length()).toBe(0);
  });
});
