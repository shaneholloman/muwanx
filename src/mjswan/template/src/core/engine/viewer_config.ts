import * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { MjData, MjModel } from 'mujoco';
import { mjcToThreeCoordinate } from '../scene/coordinate';
import { VIEWER_CONFIG_DEFAULTS } from './viewer_config_defaults';

export type ViewerConfig = {
  /** Look-at point in MuJoCo coordinates [x forward, y left, z up]. */
  lookat?: [number, number, number];
  /** Distance from the look-at point to the camera. */
  distance?: number;
  /** Vertical field of view in degrees. */
  fovy?: number;
  /** Camera elevation in degrees (negative = camera above the look-at point). */
  elevation?: number;
  /** Camera azimuth in degrees measured from the x-axis (forward) CCW. */
  azimuth?: number;
  /** Origin type for camera tracking. */
  originType?: 'AUTO' | 'WORLD' | 'ASSET_ROOT' | 'ASSET_BODY';
  /** Entity/asset name (currently unused in single-entity scenes). */
  entityName?: string;
  /** Body name to track when originType is ASSET_BODY. */
  bodyName?: string;
  /** Whether to enable reflections. */
  enableReflections?: boolean;
  /** Whether to enable shadows. */
  enableShadows?: boolean;
  /** Viewer canvas height in pixels. */
  height?: number;
  /** Viewer canvas width in pixels. */
  width?: number;
};

export type ViewerState = {
  /** Body index to track each frame, or null. */
  trackBodyId: number | null;
  /** Previous body world position used to compute per-frame delta for parallel tracking. */
  prevBodyPos: THREE.Vector3 | null;
};

/** Camera pose in spherical MuJoCo coordinates (x forward, y left, z up). */
export type CameraView = {
  lookat: [number, number, number];
  distance: number;
  azimuth: number;
  elevation: number;
  fovy: number;
};


export function computeCameraPosition(
  lookat: [number, number, number],
  distance: number,
  elevation: number,
  azimuth: number
): THREE.Vector3 {
  const el = (elevation * Math.PI) / 180;
  const az = (azimuth * Math.PI) / 180;
  const camX = lookat[0] + distance * Math.cos(el) * Math.cos(az);
  const camY = lookat[1] + distance * Math.cos(el) * Math.sin(az);
  const camZ = lookat[2] - distance * Math.sin(el);
  return mjcToThreeCoordinate([camX, camY, camZ]);
}

/**
 * Apply a ViewerConfig after a scene loads.
 *
 * Computes camera position from lookat + distance + elevation + azimuth in
 * MuJoCo coordinates (x forward, y left, z up), then converts to Three.js.
 * Returns the ViewerState that runtime.ts must keep to drive per-frame updates.
 *
 * Mirrors the pattern of createLights() in lights.ts.
 */
export function applyViewerConfig(
  config: ViewerConfig | null,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  mjModel: MjModel | null,
  mjData: MjData | null
): ViewerState {
  const state: ViewerState = { trackBodyId: null, prevBodyPos: null };
  controls.enabled = true;

  const lookat = config?.lookat ?? VIEWER_CONFIG_DEFAULTS.lookat;
  const distance = config?.distance ?? VIEWER_CONFIG_DEFAULTS.distance;
  const elevation = config?.elevation ?? VIEWER_CONFIG_DEFAULTS.elevation;
  const azimuth = config?.azimuth ?? VIEWER_CONFIG_DEFAULTS.azimuth;

  camera.fov = config?.fovy ?? VIEWER_CONFIG_DEFAULTS.fovy;
  camera.updateProjectionMatrix();
  camera.position.copy(computeCameraPosition(lookat, distance, elevation, azimuth));
  controls.target.copy(mjcToThreeCoordinate(lookat));
  controls.update();

  if (!config) return state;

  const originType = config.originType ?? 'AUTO';

  if (originType === 'ASSET_BODY' && config.bodyName && mjModel) {
    const requestedName = config.bodyName;
    const entityName = config.entityName;
    const prefixedName = entityName ? `${entityName}/${requestedName}` : null;
    for (let b = 0; b < mjModel.nbody; b++) {
      const bodyName = mjModel.body(b).name;
      if (
        bodyName === requestedName ||
        bodyName === prefixedName ||
        bodyName.endsWith(`/${requestedName}`)
      ) {
        state.trackBodyId = b;
        break;
      }
    }
    if (state.trackBodyId === null) {
      console.warn(`[Camera] bodyName: body "${config.bodyName}" not found.`);
    }
  } else if ((originType === 'AUTO' || originType === 'ASSET_ROOT') && mjModel && mjModel.nbody > 1) {
    // Track the first non-world body (body 1 is typically the floating base).
    state.trackBodyId = 1;
  }
  // WORLD: no tracking.

  if (state.trackBodyId !== null && mjData) {
    const bodyPos = mjcToThreeCoordinate(
      mjData.xpos.slice(state.trackBodyId * 3, state.trackBodyId * 3 + 3)
    );
    camera.position.add(bodyPos);
    controls.target.add(bodyPos);
    state.prevBodyPos = bodyPos;
    controls.update();
  }

  return state;
}

/**
 * A frame's horizontal delta beyond this is an episode reset teleporting the body to a new
 * spawn, not the body walking. A desktop camera should follow it — that is how a chase
 * view keeps up — but a rig carrying a viewer must not.
 */
const MAX_RIG_FOLLOW_STEP = 1.0;

/**
 * Update the Three.js camera each frame for body tracking.
 *
 * Must be called before controls.update().
 * Mirrors the pattern of updateLightsFromData() in lights.ts.
 */
export function updateCameraFromData(
  mjData: MjData,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  state: ViewerState,
  /** In XR the head pose owns the camera, so the delta moves the rig instead. */
  rig: THREE.Object3D | null = null
): void {
  if (state.trackBodyId !== null) {
    // Parallel tracking: the viewpoint and the orbit target both take the body's
    // delta each frame, so the camera angle and zoom level survive.
    const b = state.trackBodyId;
    const bodyPos = mjcToThreeCoordinate(mjData.xpos.slice(b * 3, b * 3 + 3));
    if (state.prevBodyPos !== null) {
      const delta = bodyPos.clone().sub(state.prevBodyPos);
      controls.target.add(delta);
      if (rig) {
        followWithRig(rig, delta);
      } else {
        camera.position.add(delta);
      }
    }
    state.prevBodyPos = bodyPos;
  }
}

/**
 * Horizontal only: the rig's height belongs to the ground under the viewer, not to the
 * body's own rise and fall, which on a rough terrain is a different height entirely.
 */
function followWithRig(rig: THREE.Object3D, delta: THREE.Vector3): void {
  if (Math.hypot(delta.x, delta.z) > MAX_RIG_FOLLOW_STEP) {
    return;
  }
  rig.position.x += delta.x;
  rig.position.z += delta.z;
}
