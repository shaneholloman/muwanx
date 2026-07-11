import * as THREE from 'three';
import { SplatMesh, getSplatFileType } from '@sparkjsdev/spark';
export type { SplatMesh };

/** Spherical splat placement, in the splat's own frame. Public engine input. */
export interface SplatTransform {
  scale?: number;
  xOffset?: number;
  yOffset?: number;
  zOffset?: number;
  /** Degrees, applied on top of the COLMAP→Three.js base rotation. */
  roll?: number;
  pitch?: number;
  yaw?: number;
}

export interface SplatConfig {
  name: string;
  /** Relative asset path for bundled .spz files (resolved to URL by App.tsx). */
  path?: string;
  /** External URL for non-bundled .spz files. Exactly one of path or url must be set. */
  url?: string;
  scale?: number;
  xOffset?: number;
  yOffset?: number;
  zOffset?: number;
  /** Roll in degrees applied on top of the COLMAP→Three.js base rotation. */
  roll?: number;
  /** Pitch in degrees applied on top of the COLMAP→Three.js base rotation. */
  pitch?: number;
  /** Yaw in degrees applied on top of the COLMAP→Three.js base rotation. */
  yaw?: number;
  colliderUrl?: string;
  /** If true, shows scale and offset controls in the viewer control panel. */
  control?: boolean;
}

const DEG2RAD = Math.PI / 180;
const BASE_QUAT = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI, 0, 0));

/** Apply scale, position, and rotation from a transform to an existing SplatMesh. */
export function applySplatTransform(splat: SplatMesh, transform: SplatTransform): void {
  const scale = transform.scale ?? 1.0;
  const xOffset = transform.xOffset ?? 0.0;
  const yOffset = transform.yOffset ?? 0.0;
  const zOffset = transform.zOffset ?? 0.0;
  const roll  = (transform.roll  ?? 0.0) * DEG2RAD;
  const pitch = (transform.pitch ?? 0.0) * DEG2RAD;
  const yaw   = (transform.yaw   ?? 0.0) * DEG2RAD;

  splat.scale.setScalar(scale);

  // WorldLabs splats use COLMAP/OpenCV convention (Y-down, Z-into-scene).
  // Rotating 180° around X flips to Three.js convention (Y-up, Z-towards-viewer).
  // User roll/pitch/yaw are applied on top via quaternion composition.
  const userQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll));
  splat.quaternion.copy(BASE_QUAT.clone().multiply(userQuat));

  splat.position.set(xOffset * scale, zOffset * scale, yOffset * scale);
}

/** Build a SplatMesh from raw `.spz`/`.ply`/... bytes and place it in the scene. */
export function loadSplat(
  data: ArrayBuffer,
  transform: SplatTransform,
  scene: THREE.Scene
): SplatMesh {
  const fileBytes = new Uint8Array(data);
  const splat = new SplatMesh({ fileBytes, fileType: getSplatFileType(fileBytes) });
  applySplatTransform(splat, transform);
  scene.add(splat);
  return splat;
}

export function disposeSplat(splat: SplatMesh, scene: THREE.Scene): void {
  scene.remove(splat);
  splat.dispose?.();
}
