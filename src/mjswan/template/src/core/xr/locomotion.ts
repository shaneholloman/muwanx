/**
 * Thumbstick locomotion in XR: the left stick slides the viewer, the right stick turns it.
 *
 * three.js writes the head pose into `camera.matrix` relative to the camera's *parent*, so
 * a Group between the scene and the camera is the one place an app can add movement of its
 * own. Turns pivot on the head rather than the rig origin, or turning would swing a viewer
 * standing off-centre sideways through the scene.
 */
import * as THREE from 'three';

const UP = new THREE.Vector3(0, 1, 0);

/** A resting stick still reports a little; below this it means nothing. */
const DEADZONE = 0.15;

/** Metres a second, and degrees a second at full deflection: comfort over speed. */
const MOVE_SPEED = 1.5;
const TURN_SPEED = 90;

const forward = new THREE.Vector3();
const rightward = new THREE.Vector3();
const head = new THREE.Vector3();
const rotation = new THREE.Quaternion();

/** `xr-standard` puts the thumbstick on axes 2/3. */
function thumbstick(gamepad: Gamepad): [number, number] {
  const axes = gamepad.axes;
  const [x, y] = axes.length >= 4 ? [axes[2], axes[3]] : [axes[0], axes[1]];
  return [deadzone(x ?? 0), deadzone(y ?? 0)];
}

function deadzone(value: number): number {
  return Math.abs(value) < DEADZONE ? 0 : value;
}

/** Once per rendered XR frame. */
export function updateXrLocomotion(
  rig: THREE.Object3D,
  camera: THREE.Camera,
  session: XRSession | null,
  seconds: number
): void {
  if (!session || seconds <= 0) {
    return;
  }

  let slideBy: [number, number] | null = null;
  let yaw = 0;
  for (const source of session.inputSources) {
    if (!source.gamepad) {
      continue;
    }
    const [x, y] = thumbstick(source.gamepad);
    if (source.handedness === 'right') {
      yaw = x;
    } else {
      slideBy = [x, y];
    }
  }

  if (slideBy) {
    slide(rig, camera, slideBy[0], slideBy[1], seconds);
  }
  if (yaw !== 0) {
    turn(rig, camera, yaw, seconds);
  }
}

function slide(
  rig: THREE.Object3D,
  camera: THREE.Camera,
  x: number,
  y: number,
  seconds: number
): void {
  if (x === 0 && y === 0) {
    return;
  }
  camera.getWorldDirection(forward);
  forward.y = 0;
  if (forward.lengthSq() < 1e-6) {
    return; // looking straight up or down: no heading to slide along
  }
  forward.normalize();
  rightward.crossVectors(forward, UP).normalize();
  // Stick forward is -1 on the `xr-standard` Y axis.
  rig.position.addScaledVector(forward, -y * MOVE_SPEED * seconds);
  rig.position.addScaledVector(rightward, x * MOVE_SPEED * seconds);
}

function turn(rig: THREE.Object3D, camera: THREE.Camera, yaw: number, seconds: number): void {
  // Negative about +Y turns the view to the viewer's right, matching a stick pushed right.
  const radians = (-yaw * TURN_SPEED * seconds * Math.PI) / 180;
  rotation.setFromAxisAngle(UP, radians);
  camera.getWorldPosition(head);
  rig.position.sub(head).applyQuaternion(rotation).add(head);
  // Normalized, or thousands of small products drift off unit length over a session.
  rig.quaternion.premultiply(rotation).normalize();
}
