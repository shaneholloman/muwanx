/**
 * WebXR hand tracking as bodies inside the simulation.
 *
 * A hand that only pushes forces into the scene cannot be touched back, so the
 * tracked joints go into the model itself. Mocap bodies are the cheap way in: they
 * carry no degrees of freedom, so appending them leaves `nq`, the actuators and the
 * joint order a policy reads untouched, and their pose is one write per step into
 * `mjData.mocap_pos` / `mocap_quat`.
 *
 * A mocap body alone can only shove things, though. MuJoCo takes contact velocity
 * from body velocity, and a teleported body has none, so friction never transmits
 * the hand's motion and a pinched object is left behind. Each joint is therefore a
 * pair: the mocap target this module writes, plus a dynamic sphere welded to it that
 * owns the contact. The weld carries the target's motion as real momentum, which is
 * what makes picking something up work rather than only batting it around.
 */
import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';

import { threeToMjcCoordinate, threeToMjcQuaternion } from '../scene/coordinate';

/** Wrist plus the five tips: enough to push with the palm and pinch with any finger. */
export const DEFAULT_HAND_JOINTS: readonly XRHandJoint[] = [
  'wrist',
  'thumb-tip',
  'index-finger-tip',
  'middle-finger-tip',
  'ring-finger-tip',
  'pinky-finger-tip',
];

export type HandMocapConfig = {
  joints?: readonly XRHandJoint[];
  /** Fingertip sphere radius, metres. */
  radius?: number;
  /** Fingertip mass — how hard the hand can shove something heavy. */
  mass?: number;
};

/** Where an untracked hand waits: far below any scene, out of every collision. */
const PARKED_Z = -100;

/** Spread along the parking row, so parked fingertips do not contact each other. */
const PARKED_SPACING = 0.05;

/** WebXR reports at most two hands, and `renderer.xr.getHand` indexes them. */
const HAND_COUNT = 2;

function bodyName(hand: number, joint: XRHandJoint, kind: 'target' | 'tip'): string {
  return `mjswan_xr${hand}_${joint}_${kind}`;
}

/** Derived from the indices in both places, so the XML and the writes agree. */
function parkedPosition(hand: number, jointIndex: number, jointCount: number): THREE.Vector3 {
  return new THREE.Vector3((hand * jointCount + jointIndex) * PARKED_SPACING, 0, PARKED_Z);
}

/**
 * Append the target/fingertip pairs to `xml` as a second `<worldbody>` and
 * `<equality>` — MJCF merges repeated sections, so this needs nothing from the model
 * it edits (the bodies may live entirely in `<include>`d files). Appending rather
 * than inserting is what keeps the robot's own free joint at `qpos[0]`, which
 * `PolicyStateBuilder` reads the root pose from.
 */
export function injectHandMocapXml(xml: string, config: HandMocapConfig = {}): string {
  const joints = config.joints ?? DEFAULT_HAND_JOINTS;
  const radius = config.radius ?? 0.012;
  const mass = config.mass ?? 0.05;

  const bodies: string[] = [];
  const welds: string[] = [];
  for (let hand = 0; hand < HAND_COUNT; hand++) {
    for (const [index, joint] of joints.entries()) {
      const target = bodyName(hand, joint, 'target');
      const tip = bodyName(hand, joint, 'tip');
      const { x, y, z } = parkedPosition(hand, index, joints.length);
      bodies.push(`    <body name="${target}" mocap="true" pos="${x} ${y} ${z}"/>`);
      bodies.push(
        `    <body name="${tip}" pos="${x} ${y} ${z}">\n` +
          `      <freejoint/>\n` +
          `      <geom type="sphere" size="${radius}" mass="${mass}" condim="6"` +
          ` friction="2 0.05 0.001" rgba="0.35 0.75 1 0.55"/>\n` +
          `    </body>`,
      );
      // Soft enough to stay stable at a browser timestep, stiff enough to grip.
      welds.push(
        `    <weld body1="${target}" body2="${tip}" solref="0.005 1" solimp="0.95 0.99 0.001"/>`,
      );
    }
  }

  const block =
    `  <worldbody>\n${bodies.join('\n')}\n  </worldbody>\n` +
    `  <equality>\n${welds.join('\n')}\n  </equality>\n`;

  const close = xml.lastIndexOf('</mujoco>');
  if (close < 0) {
    throw new Error('injectHandMocapXml: scene XML has no closing </mujoco>');
  }
  return `${xml.slice(0, close)}${block}${xml.slice(close)}`;
}

/** Rewrite a scene XML in the Emscripten VFS in place, ahead of `mj_loadXML`. */
export function injectHandMocapFile(
  mujoco: MainModule,
  path: string,
  config: HandMocapConfig = {},
): void {
  const xml = new TextDecoder().decode(mujoco.FS.readFile(path));
  mujoco.FS.writeFile(path, injectHandMocapXml(xml, config));
}

type BoundJoint = {
  hand: number;
  joint: XRHandJoint;
  tipId: number;
  mocapId: number;
  qposAdr: number;
  qvelAdr: number;
  parked: THREE.Vector3;
  tracked: boolean;
};

export class HandMocap {
  private readonly hands: THREE.XRHandSpace[];
  private readonly joints: readonly XRHandJoint[];
  private bound: BoundJoint[] = [];
  private readonly position = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();

  constructor(hands: THREE.XRHandSpace[], config: HandMocapConfig = {}) {
    this.hands = hands;
    this.joints = config.joints ?? DEFAULT_HAND_JOINTS;
  }

  /** Resolve the injected bodies in a freshly loaded model. */
  bind(mujoco: MainModule, mjModel: MjModel): void {
    const body = mujoco.mjtObj.mjOBJ_BODY.value;
    this.bound = [];
    for (let hand = 0; hand < this.hands.length && hand < HAND_COUNT; hand++) {
      for (const [index, joint] of this.joints.entries()) {
        const targetId = mujoco.mj_name2id(mjModel, body, bodyName(hand, joint, 'target'));
        const tipId = mujoco.mj_name2id(mjModel, body, bodyName(hand, joint, 'tip'));
        if (targetId < 0 || tipId < 0) {
          continue;
        }
        const jntAdr = mjModel.body_jntadr[tipId];
        this.bound.push({
          hand,
          joint,
          tipId,
          mocapId: mjModel.body_mocapid[targetId],
          qposAdr: mjModel.jnt_qposadr[jntAdr],
          qvelAdr: mjModel.jnt_dofadr[jntAdr],
          parked: parkedPosition(hand, index, this.joints.length),
          tracked: false,
        });
      }
    }
    if (this.bound.length === 0) {
      console.warn('[HandMocap] no injected hand bodies found in the model');
    }
  }

  /** The rendered bodies, for anything measuring the scene: parked hands are far outside it. */
  tipBodyIds(): number[] {
    return this.bound.map((bound) => bound.tipId);
  }

  /**
   * Point each mocap target at its tracked joint. A joint that has just come back
   * teleports its fingertip instead of letting the weld drag it in from the parking
   * spot, which would arrive as one enormous impulse.
   */
  update(mjData: MjData): void {
    for (const bound of this.bound) {
      const joint = this.hands[bound.hand]?.joints[bound.joint];
      if (!joint?.visible) {
        if (bound.tracked) {
          this.parkJoint(mjData, bound);
        }
        continue;
      }
      joint.getWorldPosition(this.position);
      joint.getWorldQuaternion(this.quaternion);
      const pos = threeToMjcCoordinate(this.position);
      const quat = threeToMjcQuaternion(this.quaternion);
      this.writeTarget(mjData, bound, pos, quat);
      if (!bound.tracked) {
        this.writeTip(mjData, bound, pos, quat);
        bound.tracked = true;
      }
    }
  }

  /**
   * Park every hand. Called after a sim reset: a keyframe written for the model
   * before injection is zero-padded for the appended free joints, so the reset
   * spawns the fingertips at the world origin, in the middle of the scene.
   */
  park(mjData: MjData): void {
    for (const bound of this.bound) {
      this.parkJoint(mjData, bound);
    }
  }

  private parkJoint(mjData: MjData, bound: BoundJoint): void {
    const identity: [number, number, number, number] = [1, 0, 0, 0];
    this.writeTarget(mjData, bound, bound.parked, identity);
    this.writeTip(mjData, bound, bound.parked, identity);
    bound.tracked = false;
  }

  private writeTarget(
    mjData: MjData,
    bound: BoundJoint,
    pos: THREE.Vector3,
    quat: readonly [number, number, number, number],
  ): void {
    const mocapPos = mjData.mocap_pos;
    const mocapQuat = mjData.mocap_quat;
    mocapPos[bound.mocapId * 3 + 0] = pos.x;
    mocapPos[bound.mocapId * 3 + 1] = pos.y;
    mocapPos[bound.mocapId * 3 + 2] = pos.z;
    for (let i = 0; i < 4; i++) {
      mocapQuat[bound.mocapId * 4 + i] = quat[i];
    }
  }

  private writeTip(
    mjData: MjData,
    bound: BoundJoint,
    pos: THREE.Vector3,
    quat: readonly [number, number, number, number],
  ): void {
    const qpos = mjData.qpos;
    const qvel = mjData.qvel;
    qpos[bound.qposAdr + 0] = pos.x;
    qpos[bound.qposAdr + 1] = pos.y;
    qpos[bound.qposAdr + 2] = pos.z;
    for (let i = 0; i < 4; i++) {
      qpos[bound.qposAdr + 3 + i] = quat[i];
    }
    for (let i = 0; i < 6; i++) {
      qvel[bound.qvelAdr + i] = 0;
    }
  }
}
