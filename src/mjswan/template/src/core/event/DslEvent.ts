import { EventBase, type EventConfig, type EventContext } from './EventBase';

/**
 * Declarative reset event (see ADR 0003).
 *
 * Applies a list of mutation descriptors to `mjData` on reset — joint qpos/qvel
 * offsets and free-joint root pose sampling.  The mutations are data passed in
 * `config.mutations`; this class is the only TS implementation needed.
 */

type Sample =
  | { dist: 'uniform'; low: number; high: number }
  | { dist: 'uniform_xyz'; x: [number, number]; y: [number, number]; z: [number, number] };

type Mutation = {
  target: 'joint_qpos' | 'joint_qvel' | 'freejoint_pos' | 'freejoint_yaw';
  op: 'add' | 'set' | 'compose';
  sample: Sample;
  select?: { entity_name?: string; joint_names?: string[]; joint_ids?: number[] };
  clip_to_limits?: boolean;
};

type MjModel = import('mujoco').MjModel;

function sampleUniform(low: number, high: number): number {
  return low + Math.random() * (high - low);
}

function modelJointNames(mjModel: MjModel): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let j = 0; j < mjModel.njnt; j++) {
    let start = mjModel.name_jntadr[j];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

function findFreeJoint(mjModel: MjModel): number {
  for (let j = 0; j < mjModel.njnt; j++) {
    if (mjModel.jnt_type[j] === 0) return j; // mjJNT_FREE = 0
  }
  return -1;
}

export class DslEvent extends EventBase {
  private readonly mutations: Mutation[];

  constructor(config: EventConfig & { mutations?: unknown[] }) {
    super(config);
    this.mutations = (config.mutations ?? []) as Mutation[];
  }

  onReset(context: EventContext): void {
    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return;
    for (const m of this.mutations) {
      switch (m.target) {
        case 'joint_qpos':
          this.applyJointOffset(mjModel, mjData, m, 'qpos');
          break;
        case 'joint_qvel':
          this.applyJointOffset(mjModel, mjData, m, 'qvel');
          break;
        case 'freejoint_pos':
          this.applyFreejointPos(mjModel, mjData, m);
          break;
        case 'freejoint_yaw':
          this.applyFreejointYaw(mjModel, mjData, m);
          break;
      }
    }
  }

  private resolveJointIndices(mjModel: MjModel, m: Mutation): number[] {
    const sel = m.select ?? {};
    if (sel.joint_ids && sel.joint_ids.length > 0) {
      return sel.joint_ids.filter((idx) => idx >= 0 && idx < mjModel.njnt);
    }
    const names = modelJointNames(mjModel);
    if (sel.joint_names && sel.joint_names.length > 0) {
      return sel.joint_names
        .map((target) => this.findJointIndex(names, target, sel.entity_name))
        .filter((idx): idx is number => idx !== null);
    }
    // Default: all non-free joints, optionally scoped to an entity.
    return names
      .map((name, idx) => ({ name, idx }))
      .filter(({ name, idx }) => {
        if (mjModel.jnt_type[idx] === 0) return false;
        if (!sel.entity_name) return true;
        return name === sel.entity_name || name.startsWith(`${sel.entity_name}/`);
      })
      .map(({ idx }) => idx);
  }

  private findJointIndex(
    names: string[],
    target: string,
    entityName?: string,
  ): number | null {
    const exact = names.indexOf(target);
    if (exact >= 0) return exact;
    if (entityName) {
      const qualified = names.indexOf(`${entityName}/${target}`);
      if (qualified >= 0) return qualified;
    }
    const suffix = names
      .map((name, idx) => ({ name, idx }))
      .filter(({ name }) => name === target || name.endsWith(`/${target}`));
    if (suffix.length === 1) return suffix[0].idx;
    if (entityName) {
      const scoped = suffix.find(
        ({ name }) => name === entityName || name.startsWith(`${entityName}/`),
      );
      if (scoped) return scoped.idx;
    }
    return null;
  }

  private applyJointOffset(
    mjModel: MjModel,
    mjData: import('mujoco').MjData,
    m: Mutation,
    kind: 'qpos' | 'qvel',
  ): void {
    if (m.sample.dist !== 'uniform') return;
    const { low, high } = m.sample;
    for (const jointIdx of this.resolveJointIndices(mjModel, m)) {
      const jointType = mjModel.jnt_type[jointIdx];
      if (jointType !== 2 && jointType !== 3) continue; // hinge=3, slide=2
      if (kind === 'qpos') {
        const adr = mjModel.jnt_qposadr[jointIdx];
        mjData.qpos[adr] += sampleUniform(low, high);
        if (m.clip_to_limits && mjModel.jnt_limited[jointIdx]) {
          const lo = mjModel.jnt_range[jointIdx * 2];
          const hi = mjModel.jnt_range[jointIdx * 2 + 1];
          mjData.qpos[adr] = Math.min(Math.max(mjData.qpos[adr], lo), hi);
        }
      } else {
        mjData.qvel[mjModel.jnt_dofadr[jointIdx]] += sampleUniform(low, high);
      }
    }
  }

  private applyFreejointPos(
    mjModel: MjModel,
    mjData: import('mujoco').MjData,
    m: Mutation,
  ): void {
    if (m.sample.dist !== 'uniform_xyz') return;
    const j = findFreeJoint(mjModel);
    if (j < 0) return;
    const adr = mjModel.jnt_qposadr[j];
    mjData.qpos[adr] += sampleUniform(m.sample.x[0], m.sample.x[1]);
    mjData.qpos[adr + 1] += sampleUniform(m.sample.y[0], m.sample.y[1]);
    mjData.qpos[adr + 2] += sampleUniform(m.sample.z[0], m.sample.z[1]);
  }

  private applyFreejointYaw(
    mjModel: MjModel,
    mjData: import('mujoco').MjData,
    m: Mutation,
  ): void {
    if (m.sample.dist !== 'uniform') return;
    const j = findFreeJoint(mjModel);
    if (j < 0) return;
    const quatAdr = mjModel.jnt_qposadr[j] + 3;
    const yaw = sampleUniform(m.sample.low, m.sample.high);
    const hw = Math.cos(yaw / 2);
    const hz = Math.sin(yaw / 2);
    const w = mjData.qpos[quatAdr];
    const x = mjData.qpos[quatAdr + 1];
    const y = mjData.qpos[quatAdr + 2];
    const z = mjData.qpos[quatAdr + 3];
    // delta (hw, 0, 0, hz) composed on the left, matching ResetRootStateUniform.
    mjData.qpos[quatAdr] = hw * w - hz * z;
    mjData.qpos[quatAdr + 1] = hw * x + hz * y;
    mjData.qpos[quatAdr + 2] = hw * y - hz * x;
    mjData.qpos[quatAdr + 3] = hw * z + hz * w;
  }
}
