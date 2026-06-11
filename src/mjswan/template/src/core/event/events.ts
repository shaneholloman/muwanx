import { EventBase, type EventConfig, type EventContext } from './EventBase';
import { CustomEvents } from './custom_events';

/**
 * Reset root state with uniform random pose sampling.
 *
 * Applies random offsets to the free joint root position and orientation.
 * Mirrors mjlab's reset_root_state_uniform.
 *
 * Params:
 *   pose_range: { x?, y?, z?, roll?, pitch?, yaw? } — each a [min, max] tuple
 */
export class ResetRootStateUniform extends EventBase {
  private poseRange: Record<string, [number, number]>;

  constructor(config: EventConfig) {
    super(config);
    this.poseRange = (config.params?.pose_range as Record<string, [number, number]>) ?? {};
  }

  onReset(context: EventContext): void {
    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return;

    // Find the free joint (root body)
    const freeJointIdx = this._findFreeJoint(mjModel);
    if (freeJointIdx === -1) return;

    const qposAdr = mjModel.jnt_qposadr[freeJointIdx];

    const sample = (key: string): number => {
      const range = this.poseRange[key];
      if (!range) return 0;
      return range[0] + Math.random() * (range[1] - range[0]);
    };

    // Apply x/y/z offset
    mjData.qpos[qposAdr + 0] += sample('x');
    mjData.qpos[qposAdr + 1] += sample('y');
    mjData.qpos[qposAdr + 2] += sample('z');

    // Apply yaw rotation (compose with existing quaternion)
    const yaw = sample('yaw');
    if (yaw !== 0) {
      this._applyYawRotation(mjData.qpos, qposAdr + 3, yaw);
    }
  }

  private _findFreeJoint(mjModel: import('mujoco').MjModel): number {
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_type[i] === 0) return i; // mjJNT_FREE = 0
    }
    return -1;
  }

  private _applyYawRotation(
    qpos: Float64Array,
    quatAdr: number,
    yaw: number
  ): void {
    const hw = Math.cos(yaw / 2);
    const hz = Math.sin(yaw / 2);
    // delta quaternion: (hw, 0, 0, hz)
    const w = qpos[quatAdr + 0];
    const x = qpos[quatAdr + 1];
    const y = qpos[quatAdr + 2];
    const z = qpos[quatAdr + 3];
    qpos[quatAdr + 0] = hw * w - hz * z;
    qpos[quatAdr + 1] = hw * x + hz * y;  // Note: MuJoCo quat is (w, x, y, z)
    qpos[quatAdr + 2] = hw * y - hz * x;
    qpos[quatAdr + 3] = hw * z + hz * w;
  }
}

// ResetRootStateFromFlatPatches is NOT a core built-in — it reads terrain
// flat-patch data (an engine capability, and a mjswan browser enhancement, not
// an mjlab term).  Tasks that want patch-based spawning provide it task-side
// via ts_src (see examples/mjlab/defaults/events/ResetRootStateFromFlatPatches.ts)
// and register it with register_event_func.  See ADR 0003.

type ScalarRange = [number, number];

function normalizeRange(value: unknown): ScalarRange {
  if (
    Array.isArray(value)
    && value.length >= 2
    && typeof value[0] === 'number'
    && typeof value[1] === 'number'
  ) {
    return [value[0], value[1]];
  }
  return [0, 0];
}

function sampleRange(range: ScalarRange): number {
  const [min, max] = range;
  return min + Math.random() * (max - min);
}

function getModelJointNames(mjModel: import('mujoco').MjModel): string[] {
  const namesArray = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let jointIdx = 0; jointIdx < mjModel.njnt; jointIdx++) {
    let start = mjModel.name_jntadr[jointIdx];
    let end = start;
    while (end < namesArray.length && namesArray[end] !== 0) end++;
    names.push(decoder.decode(namesArray.subarray(start, end)));
  }
  return names;
}

/**
 * No-op event preserved for mjlab config compatibility.
 *
 * mjlab's randomize_terrain mutates terrain state during reset; in mjswan the
 * terrain mesh is baked into the exported MuJoCo scene, so there is no
 * browser-side terrain generator to resample.  Kept as an explicit built-in
 * so configs that reference it do not warn.
 */
export class RandomizeTerrain extends EventBase {
  constructor(config: EventConfig) {
    super(config);
  }

  onReset(_context: EventContext): void {}
}

/**
 * Reset selected joints by adding a uniform-random offset within
 * `position_range` and `velocity_range`.  Honours joint position limits.
 *
 * Params:
 *   entity_name?    — restrict to joints under `<entity_name>/...`.
 *   joint_names?    — explicit joint name list (entity-resolved if needed).
 *   joint_ids?      — explicit joint index list (bypasses name resolution).
 *   position_range  — [min, max] uniform offset for qpos.
 *   velocity_range  — [min, max] uniform offset for qvel.
 *
 * mjlab: events.reset_joints_by_offset
 */
export class ResetJointsByOffset extends EventBase {
  private readonly entityName: string | null;
  private readonly jointNames: string[] | null;
  private readonly jointIds: number[] | null;
  private readonly positionRange: ScalarRange;
  private readonly velocityRange: ScalarRange;

  constructor(config: EventConfig) {
    super(config);
    this.entityName = typeof config.params?.entity_name === 'string'
      ? config.params.entity_name
      : null;
    this.jointNames = Array.isArray(config.params?.joint_names)
      ? config.params!.joint_names.filter((v): v is string => typeof v === 'string')
      : null;
    this.jointIds = Array.isArray(config.params?.joint_ids)
      ? config.params!.joint_ids.filter((v): v is number => typeof v === 'number')
      : null;
    this.positionRange = normalizeRange(config.params?.position_range);
    this.velocityRange = normalizeRange(config.params?.velocity_range);
  }

  onReset(context: EventContext): void {
    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return;

    const jointIndices = this.resolveJointIndices(mjModel);
    for (const jointIdx of jointIndices) {
      const jointType = mjModel.jnt_type[jointIdx];
      if (jointType !== 2 && jointType !== 3) continue;

      const qposAdr = mjModel.jnt_qposadr[jointIdx];
      const qvelAdr = mjModel.jnt_dofadr[jointIdx];
      mjData.qpos[qposAdr] += sampleRange(this.positionRange);
      mjData.qvel[qvelAdr] += sampleRange(this.velocityRange);

      if (mjModel.jnt_limited[jointIdx]) {
        const rangeAdr = jointIdx * 2;
        const lower = mjModel.jnt_range[rangeAdr];
        const upper = mjModel.jnt_range[rangeAdr + 1];
        mjData.qpos[qposAdr] = Math.min(Math.max(mjData.qpos[qposAdr], lower), upper);
      }
    }
  }

  private resolveJointIndices(mjModel: import('mujoco').MjModel): number[] {
    if (this.jointIds && this.jointIds.length > 0) {
      return this.jointIds.filter((idx) => idx >= 0 && idx < mjModel.njnt);
    }

    const modelJointNames = getModelJointNames(mjModel);
    if (this.jointNames && this.jointNames.length > 0) {
      return this.jointNames
        .map((name) => this.findJointIndex(modelJointNames, name))
        .filter((idx): idx is number => idx !== null);
    }

    return modelJointNames
      .map((name, idx) => ({ name, idx }))
      .filter(({ name, idx }) => {
        if (mjModel.jnt_type[idx] === 0) return false;
        if (!this.entityName) return true;
        return name === this.entityName || name.startsWith(`${this.entityName}/`);
      })
      .map(({ idx }) => idx);
  }

  private findJointIndex(modelJointNames: string[], targetName: string): number | null {
    const exactIdx = modelJointNames.indexOf(targetName);
    if (exactIdx >= 0) return exactIdx;

    if (this.entityName) {
      const qualified = `${this.entityName}/${targetName}`;
      const qualifiedIdx = modelJointNames.indexOf(qualified);
      if (qualifiedIdx >= 0) return qualifiedIdx;
    }

    const suffixMatches = modelJointNames
      .map((name, idx) => ({ name, idx }))
      .filter(({ name }) => name === targetName || name.endsWith(`/${targetName}`));
    if (suffixMatches.length === 1) return suffixMatches[0].idx;
    if (this.entityName) {
      const scoped = suffixMatches.find(
        ({ name }) => name === this.entityName || name.startsWith(`${this.entityName}/`),
      );
      if (scoped) return scoped.idx;
    }
    return null;
  }
}

export type { EventConstructor } from './EventBase';

const BuiltinEvents: Record<string, import('./EventBase').EventConstructor> = {
  ResetRootStateUniform,
  RandomizeTerrain,
  ResetJointsByOffset,
};

export const Events: Record<string, import('./EventBase').EventConstructor> = {
  ...BuiltinEvents,
  ...CustomEvents,
};
