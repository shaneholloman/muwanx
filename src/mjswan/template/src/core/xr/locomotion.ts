/**
 * Thumbstick locomotion in XR: the left stick slides the viewer, the right stick turns it.
 *
 * three.js writes the head pose into `camera.matrix` relative to the camera's *parent*,
 * so a Group between the scene and the camera is the one place an app can add movement of
 * its own. That rig carries the tracked hands too, or moving would leave them behind.
 *
 * Turns are snapped rather than smooth, and pivot on the head rather than the rig origin:
 * continuous yaw is the classic nausea trigger, and turning about the origin would swing
 * a viewer standing off-centre sideways through the scene.
 */
import * as THREE from 'three';

const UP = new THREE.Vector3(0, 1, 0);

/** A resting stick still reports a little; below this it means nothing. */
const DEADZONE = 0.15;

/** Metres per second, and degrees per snap: comfort over speed. */
const MOVE_SPEED = 1.5;
const SNAP_DEGREES = 30;

/** Past this a turn fires, and it re-arms below it, so one push is one turn. */
const SNAP_THRESHOLD = 0.7;

/** A tab away leaves a huge gap between frames; unclamped, the first one back teleports. */
const MAX_FRAME_SECONDS = 0.1;

/** `xr-standard` puts the thumbstick on axes 2/3, leaving 0/1 to a touchpad if there is one. */
function thumbstick(gamepad: Gamepad): [number, number] {
  const axes = gamepad.axes;
  const [x, y] = axes.length >= 4 ? [axes[2], axes[3]] : [axes[0], axes[1]];
  return [deadzone(x ?? 0), deadzone(y ?? 0)];
}

function deadzone(value: number): number {
  return Math.abs(value) < DEADZONE ? 0 : value;
}

export class XrLocomotion {
  private readonly rig: THREE.Object3D;
  private readonly camera: THREE.Camera;
  private lastTime: number | null = null;
  private snapArmed = true;
  private readonly forward = new THREE.Vector3();
  private readonly right = new THREE.Vector3();
  private readonly head = new THREE.Vector3();
  private readonly turn = new THREE.Quaternion();

  constructor(rig: THREE.Object3D, camera: THREE.Camera) {
    this.rig = rig;
    this.camera = camera;
  }

  /** Once per rendered XR frame. A session whose controllers report no gamepad is a no-op. */
  update(session: XRSession | null, now: number = performance.now()): void {
    if (!session) {
      this.lastTime = null;
      return;
    }
    const seconds =
      this.lastTime === null ? 0 : Math.min((now - this.lastTime) / 1000, MAX_FRAME_SECONDS);
    this.lastTime = now;

    let slide: [number, number] | null = null;
    let yaw = 0;
    for (const source of Array.from(session.inputSources)) {
      if (!source.gamepad) {
        continue;
      }
      const [x, y] = thumbstick(source.gamepad);
      if (source.handedness === 'right') {
        yaw = x;
      } else {
        slide = [x, y];
      }
    }

    if (slide && seconds > 0) {
      this.slide(slide[0], slide[1], seconds);
    }
    this.snapTurn(yaw);
  }

  /** Back to the world origin, so leaving XR hands the desktop camera back where it was. */
  reset(): void {
    this.rig.position.set(0, 0, 0);
    this.rig.quaternion.identity();
    this.lastTime = null;
    this.snapArmed = true;
  }

  private slide(x: number, y: number, seconds: number): void {
    if (x === 0 && y === 0) {
      return;
    }
    this.camera.getWorldDirection(this.forward);
    this.forward.y = 0;
    if (this.forward.lengthSq() < 1e-6) {
      return; // looking straight up or down: no heading to slide along
    }
    this.forward.normalize();
    this.right.crossVectors(this.forward, UP).normalize();
    // Stick forward is -1 on the `xr-standard` Y axis.
    this.rig.position.addScaledVector(this.forward, -y * MOVE_SPEED * seconds);
    this.rig.position.addScaledVector(this.right, x * MOVE_SPEED * seconds);
  }

  private snapTurn(yaw: number): void {
    if (Math.abs(yaw) < SNAP_THRESHOLD) {
      this.snapArmed = true;
      return;
    }
    if (!this.snapArmed) {
      return;
    }
    this.snapArmed = false;

    // Negative about +Y turns the view to the viewer's right, matching a stick pushed right.
    this.turn.setFromAxisAngle(UP, (-Math.sign(yaw) * SNAP_DEGREES * Math.PI) / 180);
    this.camera.getWorldPosition(this.head);
    this.rig.position.sub(this.head).applyQuaternion(this.turn).add(this.head);
    this.rig.quaternion.premultiply(this.turn);
  }
}
