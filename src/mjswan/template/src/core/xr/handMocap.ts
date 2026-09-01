/**
 * WebXR hand tracking as bodies inside the simulation, so a hand can be touched back
 * rather than only shoving forces in.
 *
 * Each tracked joint is a pair: a mocap target this module writes, carrying no degrees
 * of freedom so that appending it leaves `nq`, the actuators and the joint order a
 * policy reads untouched, plus a dynamic sphere welded to it that owns the contact.
 * The weld is what makes grasping work. MuJoCo takes contact velocity from body
 * velocity and a teleported mocap body has none, so friction alone never transmits the
 * hand's motion and a pinched object is left behind.
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

/** Metres and kg; the mass is how hard the hand can shove something heavy. */
const TIP_RADIUS = 0.012;
const TIP_MASS = 0.05;

/** Where an untracked hand waits: far below any scene, out of every collision. */
const PARKED_Z = -100;

/** Spread along the parking row, so parked fingertips do not contact each other. */
const PARKED_SPACING = 0.05;

const HAND_COUNT = 2;

const IDENTITY_QUAT: readonly [number, number, number, number] = [1, 0, 0, 0];

function bodyName(hand: number, joint: XRHandJoint, kind: 'target' | 'tip'): string {
  return `mjswan_xr${hand}_${joint}_${kind}`;
}

/** Derived from the indices in both places, so the XML and the writes agree. */
function parkedPosition(hand: number, jointIndex: number, jointCount: number): THREE.Vector3 {
  return new THREE.Vector3((hand * jointCount + jointIndex) * PARKED_SPACING, 0, PARKED_Z);
}

/**
 * Append the target/fingertip pairs as a second `<worldbody>` and `<equality>`. MJCF
 * merges repeated sections, so this needs nothing from the model it edits: the bodies
 * may live entirely in `<include>`d files. Appended rather than inserted, so the
 * robot's own free joint stays at `qpos[0]`, where `PolicyStateBuilder` reads it.
 */
export function injectHandMocapXml(xml: string): string {
  const bodies: string[] = [];
  const welds: string[] = [];
  for (let hand = 0; hand < HAND_COUNT; hand++) {
    for (const [index, joint] of DEFAULT_HAND_JOINTS.entries()) {
      const target = bodyName(hand, joint, 'target');
      const tip = bodyName(hand, joint, 'tip');
      const { x, y, z } = parkedPosition(hand, index, DEFAULT_HAND_JOINTS.length);
      bodies.push(`    <body name="${target}" mocap="true" pos="${x} ${y} ${z}"/>`);
      bodies.push(
        `    <body name="${tip}" pos="${x} ${y} ${z}">\n` +
          `      <freejoint/>\n` +
          `      <geom type="sphere" size="${TIP_RADIUS}" mass="${TIP_MASS}" condim="6"` +
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
export function injectHandMocapFile(mujoco: MainModule, path: string): void {
  const xml = new TextDecoder().decode(mujoco.FS.readFile(path));
  mujoco.FS.writeFile(path, injectHandMocapXml(xml));
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
  private bound: BoundJoint[] = [];
  private readonly position = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();

  constructor(hands: THREE.XRHandSpace[]) {
    this.hands = hands;
  }

  bind(mujoco: MainModule, mjModel: MjModel): void {
    const body = mujoco.mjtObj.mjOBJ_BODY.value;
    this.bound = [];
    for (const [hand] of this.hands.entries()) {
      for (const [index, joint] of DEFAULT_HAND_JOINTS.entries()) {
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
          parked: parkedPosition(hand, index, DEFAULT_HAND_JOINTS.length),
          tracked: false,
        });
      }
    }
    if (this.bound.length === 0) {
      console.warn('[HandMocap] no injected hand bodies found in the model');
    }
  }

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

  park(mjData: MjData): void {
    for (const bound of this.bound) {
      this.parkJoint(mjData, bound);
    }
  }

  private parkJoint(mjData: MjData, bound: BoundJoint): void {
    this.writeTarget(mjData, bound, bound.parked, IDENTITY_QUAT);
    this.writeTip(mjData, bound, bound.parked, IDENTITY_QUAT);
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
