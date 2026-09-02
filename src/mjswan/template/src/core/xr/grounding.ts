/**
 * Stands the XR viewer on the ground under it, rather than on the plane z = 0.
 *
 * The rig's height used to come from two places that know nothing about the terrain: it
 * starts at z = 0, which is the terrain generator's base plane rather than its surface,
 * and body tracking added the robot's own vertical motion. On a generated `Rough` terrain
 * that is enough to put the head under the surface — MuJoCo's meshes are front-faced, so
 * the ground does not go black, it disappears and the scene shows through it.
 *
 * So the vertical is taken from a single `mj_ray` cast straight down under the head, and
 * the horizontal is left to locomotion and tracking. `EYE_HEIGHT` then fixes where the
 * eyes sit above that surface, whatever floor the headset thinks it has.
 */
import * as THREE from 'three';

import { threeToMjcCoordinate } from '../scene/coordinate';
import { geomGroupMask } from '../onnx/raycast';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;

/** Eyes this far above the ground once settled, in metres. */
const EYE_HEIGHT = 1.7;

/**
 * Group 0 only, which is where a terrain's geoms live and a robot's do not — the same
 * mask mjlab's height scan uses, and for the same reason. A scene that leaves everything
 * in group 0 can have a ray land on the robot instead; the rate limit below bounds what
 * that costs, and it corrects itself as soon as the viewer moves.
 */
const TERRAIN_GROUP = geomGroupMask([0]);

/** Straight down in MuJoCo's frame, where z is up. */
const DOWN: number[] = [0, 0, -1];

/** The ray starts this far above the head, so it never begins inside the ground. */
const RAY_START_ABOVE = 20;

/** Metres a second the floor may travel. A step edge would otherwise be a teleport. */
const CLIMB_SPEED = 3;

/** The head is never allowed closer to the ground than this, whatever the rate limit says. */
const MIN_HEAD_CLEARANCE = 0.15;

export class RigGrounding {
  private readonly rig: THREE.Object3D;
  private readonly camera: THREE.Camera;
  private readonly head = new THREE.Vector3();
  private readonly origin: number[] = [0, 0, 0];
  /** `mj_ray` writes the geom it hit here; unused, but the binding wants the slot. */
  private readonly geomId = new Int32Array(1);
  /**
   * `EYE_HEIGHT` minus the headset's own head height, sampled once per session. Fixed
   * rather than re-read, or correcting it every frame would cancel the viewer's own
   * crouching and every other head movement with it.
   */
  private eyeOffset: number | null = null;

  constructor(rig: THREE.Object3D, camera: THREE.Camera) {
    this.rig = rig;
    this.camera = camera;
  }

  /** Once per rendered XR frame, after locomotion and tracking have moved the rig. */
  update(
    mujoco: MainModule,
    mjModel: MjModel | null,
    mjData: MjData | null,
    seconds: number
  ): void {
    if (!mjModel || !mjData) {
      return;
    }
    this.camera.getWorldPosition(this.head);
    const headAboveRig = this.head.y - this.rig.position.y;
    if (this.eyeOffset === null) {
      this.eyeOffset = EYE_HEIGHT - headAboveRig;
    }

    const ground = this.sampleGround(mujoco, mjModel, mjData);
    if (ground === null) {
      return; // nothing under the viewer: keep the height it already had
    }

    const target = ground + this.eyeOffset;
    const limit = CLIMB_SPEED * seconds;
    this.rig.position.y += THREE.MathUtils.clamp(target - this.rig.position.y, -limit, limit);

    // Being inside the ground is worse than a jump, so this one ignores the rate limit.
    const floor = ground + MIN_HEAD_CLEARANCE - headAboveRig;
    if (this.rig.position.y < floor) {
      this.rig.position.y = floor;
    }
  }

  /** The session's head height is the next session's to measure again. */
  reset(): void {
    this.eyeOffset = null;
  }

  /** World height of the ground under the head, or null where the ray hit nothing. */
  private sampleGround(mujoco: MainModule, mjModel: MjModel, mjData: MjData): number | null {
    const mjc = threeToMjcCoordinate(this.head);
    this.origin[0] = mjc.x;
    this.origin[1] = mjc.y;
    this.origin[2] = mjc.z + RAY_START_ABOVE;
    const distance = mujoco.mj_ray(
      mjModel,
      mjData,
      this.origin,
      DOWN,
      TERRAIN_GROUP as unknown as number[],
      1,
      -1,
      this.geomId,
      null
    );
    if (distance < 0) {
      return null;
    }
    // MuJoCo's z is three.js's y, so the hit height needs no conversion of its own.
    return this.origin[2] - distance;
  }
}
