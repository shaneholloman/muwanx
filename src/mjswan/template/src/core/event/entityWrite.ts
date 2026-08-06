/**
 * Writes an already-computed `entity_write` value to mjData. Apply-only: the term body
 * sampled and computed inside its graph, so nothing here draws a random number.
 *
 * The kinds mirror `mjswan.compile.tracer._WRITE_FIELDS`.
 */

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

export type WriteKind = 'joint_state' | 'root_pose' | 'root_velocity';

export interface WriteTarget {
  kind: WriteKind;
  /** Entity the write applies to (informational at N=1, single-entity scenes). */
  entity?: string | null;
  fields: string[];
  /** Resolved joint indices for `joint_state`; `"all"` means every joint. */
  joint_ids?: number[] | 'all';
}

/** Graph outputs keyed by the tracer's `"<kind>__<field>"` naming. */
export type WriteValues = Record<string, Float32Array | Float64Array | number[]>;

export function decodeJointNames(mjModel: MjModel): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let j = 0; j < mjModel.njnt; j++) {
    const start = mjModel.name_jntadr[j];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

export function findFreeJoint(mjModel: MjModel): number {
  for (let j = 0; j < mjModel.njnt; j++) {
    if (mjModel.jnt_type[j] === 0) return j; // mjJNT_FREE
  }
  return -1;
}

/**
 * Apply one write target to `mjData`, returning whether anything was written. False for a
 * model with no such target, so a scene mismatch degrades rather than throwing.
 */
export function applyEntityWrite(
  mjModel: MjModel,
  mjData: MjData,
  target: WriteTarget,
  values: WriteValues,
): boolean {
  switch (target.kind) {
    case 'joint_state':
      return writeJointState(mjModel, mjData, target, values);
    case 'root_pose':
      return writeRootPose(mjModel, mjData, values['root_pose__pose']);
    case 'root_velocity':
      return writeRootVelocity(mjModel, mjData, values['root_velocity__velocity']);
    default:
      return false;
  }
}

/** Apply every write target a term emitted, in order. */
export function applyEntityWrites(
  mjModel: MjModel,
  mjData: MjData,
  targets: readonly WriteTarget[],
  values: WriteValues,
): number {
  let applied = 0;
  for (const target of targets) {
    if (applyEntityWrite(mjModel, mjData, target, values)) applied++;
  }
  return applied;
}

function resolveJointIds(mjModel: MjModel, target: WriteTarget): number[] {
  const ids = target.joint_ids;
  if (ids === undefined || ids === 'all') {
    return Array.from({ length: mjModel.njnt }, (_, j) => j);
  }
  return ids;
}

function writeJointState(
  mjModel: MjModel,
  mjData: MjData,
  target: WriteTarget,
  values: WriteValues,
): boolean {
  const position = values['joint_state__position'];
  const velocity = values['joint_state__velocity'];
  if (!position && !velocity) return false;
  const jointIds = resolveJointIds(mjModel, target);
  for (let i = 0; i < jointIds.length; i++) {
    const j = jointIds[i];
    if (j < 0 || j >= mjModel.njnt) continue;
    if (position && i < position.length) {
      mjData.qpos[mjModel.jnt_qposadr[j]] = position[i];
    }
    if (velocity && i < velocity.length) {
      mjData.qvel[mjModel.jnt_dofadr[j]] = velocity[i];
    }
  }
  return true;
}

function writeRootPose(
  mjModel: MjModel,
  mjData: MjData,
  pose: WriteValues[string] | undefined,
): boolean {
  if (!pose || pose.length < 7) return false;
  const j = findFreeJoint(mjModel);
  if (j < 0) return false;
  const adr = mjModel.jnt_qposadr[j];
  for (let i = 0; i < 7; i++) mjData.qpos[adr + i] = pose[i];
  return true;
}

function writeRootVelocity(
  mjModel: MjModel,
  mjData: MjData,
  velocity: WriteValues[string] | undefined,
): boolean {
  if (!velocity || velocity.length < 6) return false;
  const j = findFreeJoint(mjModel);
  if (j < 0) return false;
  const adr = mjModel.jnt_dofadr[j];
  for (let i = 0; i < 6; i++) mjData.qvel[adr + i] = velocity[i];
  return true;
}
