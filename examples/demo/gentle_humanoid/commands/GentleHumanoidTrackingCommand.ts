import * as THREE from 'three';

import { getPosition, getQuaternion } from 'mjswan/scene';
import { type NpzEntry, loadNpz } from 'mjswan/npz';
import { type Bytes, resolveBytes } from 'mjswan/bytes';
import type { CommandConfigEntry, CommandTerm, CommandTermContext, CommandUiConfig } from 'mjswan/command';
type GentleHumanoidMotionConfig = {
  name: string;
  // Raw .npz bytes (or a lazy loader) the app feeds via the 'motion' command's
  // motions[] — no fetch (ADR 0004 §4). Was a fetched `path` before the refactor.
  data: Bytes;
  fps?: number;
  dataset_joint_names?: string[];
  default?: boolean;
  metadata?: {
    start?: number;
    end?: number;
  };
};

type MotionFrames = {
  jointPos: Float32Array[];
  rootPos: Float32Array[];
  rootQuat: Float32Array[];
  fps: number;
  frameCount: number;
};

type RefState = {
  jointPos: Float32Array;
  rootPos: Float32Array;
  rootQuat: Float32Array;
};

function splitFrames(entry: NpzEntry): Float32Array[] {
  const totalFrames = entry.shape[0] ?? 0;
  const width = entry.shape.length <= 1 ? 1 : entry.shape.slice(1).reduce((acc, v) => acc * v, 1);
  const frames: Float32Array[] = [];
  for (let i = 0; i < totalFrames; i++) {
    const out = new Float32Array(width);
    const start = i * width;
    for (let j = 0; j < width; j++) {
      out[j] = entry.data[start + j] ?? 0.0;
    }
    frames.push(out);
  }
  return frames;
}

function readScalar(entry: NpzEntry | undefined, fallback: number): number {
  const value = entry?.data[0];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}


function normalizeQuat(q: ArrayLike<number>): Float32Array {
  const w = q[0] ?? 1.0;
  const x = q[1] ?? 0.0;
  const y = q[2] ?? 0.0;
  const z = q[3] ?? 0.0;
  const n = Math.hypot(w, x, y, z) || 1.0;
  return new Float32Array([w / n, x / n, y / n, z / n]);
}

function quatInverse(q: ArrayLike<number>): Float32Array {
  const w = q[0] ?? 1.0;
  const x = q[1] ?? 0.0;
  const y = q[2] ?? 0.0;
  const z = q[3] ?? 0.0;
  const n2 = w * w + x * x + y * y + z * z || 1.0;
  return new Float32Array([w / n2, -x / n2, -y / n2, -z / n2]);
}

function quatMultiply(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const aw = a[0] ?? 1.0;
  const ax = a[1] ?? 0.0;
  const ay = a[2] ?? 0.0;
  const az = a[3] ?? 0.0;
  const bw = b[0] ?? 1.0;
  const bx = b[1] ?? 0.0;
  const by = b[2] ?? 0.0;
  const bz = b[3] ?? 0.0;
  return new Float32Array([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ]);
}

function quatApply(q: ArrayLike<number>, v: ArrayLike<number>): Float32Array {
  const pure = new Float32Array([0.0, v[0] ?? 0.0, v[1] ?? 0.0, v[2] ?? 0.0]);
  const rotated = quatMultiply(quatMultiply(q, pure), quatInverse(q));
  return new Float32Array([rotated[1], rotated[2], rotated[3]]);
}

function yawQuat(q: ArrayLike<number>): Float32Array {
  const w = q[0] ?? 1.0;
  const x = q[1] ?? 0.0;
  const y = q[2] ?? 0.0;
  const z = q[3] ?? 0.0;
  const yaw = Math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  const half = yaw * 0.5;
  return normalizeQuat([Math.cos(half), 0.0, 0.0, Math.sin(half)]);
}

function slerpQuat(a: ArrayLike<number>, b: ArrayLike<number>, t: number): Float32Array {
  const qa = normalizeQuat(a);
  let qb = normalizeQuat(b);
  let dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
  if (dot < 0.0) {
    qb = new Float32Array([-qb[0], -qb[1], -qb[2], -qb[3]]);
    dot = -dot;
  }
  if (dot > 0.9995) {
    return normalizeQuat([
      qa[0] + t * (qb[0] - qa[0]),
      qa[1] + t * (qb[1] - qa[1]),
      qa[2] + t * (qb[2] - qa[2]),
      qa[3] + t * (qb[3] - qa[3]),
    ]);
  }
  const theta0 = Math.acos(Math.max(-1.0, Math.min(1.0, dot)));
  const theta = theta0 * t;
  const sinTheta = Math.sin(theta);
  const sinTheta0 = Math.sin(theta0);
  const s0 = Math.cos(theta) - dot * sinTheta / sinTheta0;
  const s1 = sinTheta / sinTheta0;
  return normalizeQuat([
    s0 * qa[0] + s1 * qb[0],
    s0 * qa[1] + s1 * qb[1],
    s0 * qa[2] + s1 * qb[2],
    s0 * qa[3] + s1 * qb[3],
  ]);
}

function remapJointFrames(
  frames: Float32Array[],
  sourceJointNames: string[],
  targetJointNames: string[],
): Float32Array[] {
  const sourceIndex = new Map(sourceJointNames.map((name, index) => [name, index]));
  return frames.map((frame) => {
    const out = new Float32Array(targetJointNames.length);
    for (let i = 0; i < targetJointNames.length; i++) {
      const source = sourceIndex.get(targetJointNames[i]);
      out[i] = source !== undefined ? frame[source] ?? 0.0 : 0.0;
    }
    return out;
  });
}

function sliceFrames(frames: Float32Array[], start: number, end: number): Float32Array[] {
  const first = Math.max(0, Math.floor(start));
  const last = end < 0 ? frames.length : Math.min(frames.length, Math.floor(end));
  return frames.slice(first, Math.max(first, last));
}

function setGhostMaterial(material: THREE.Material): THREE.Material {
  const next = material.clone();
  if ('transparent' in next) {
    next.transparent = true;
  }
  if ('opacity' in next) {
    next.opacity = 0.5;
  }
  if ('depthWrite' in next) {
    next.depthWrite = false;
  }
  if ('color' in next && next.color instanceof THREE.Color) {
    next.color = new THREE.Color(0.45, 0.75, 0.55);
  }
  return next;
}

function hasRenderableMesh(object: THREE.Object3D): boolean {
  let found = false;
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      found = true;
    }
  });
  return found;
}

export class GentleHumanoidTrackingCommand implements CommandTerm {
  private readonly context: CommandTermContext;
  private readonly motions: GentleHumanoidMotionConfig[];
  // The demo has a small fixed motion set; keep loaded clips cached to avoid repeated NPZ fetch/parse work.
  private readonly loadedMotions = new Map<string, MotionFrames>();
  private readonly targetJointNames: string[];
  private readonly qposAdr: number[];
  private readonly qvelAdr: number[];
  private readonly rootQposAdr: number;
  private readonly rootQvelAdr: number;
  private readonly ghostRoot: THREE.Group | null;
  private readonly ghostBodies: Map<number, THREE.Group>;
  private readonly ghostData: import('mujoco').MjData | null;
  private readonly transitionSteps: number;
  private readonly futureHistoryLen: number;
  private readonly switchTailKeepSteps: number;
  private readonly refMaxLen: number;
  private frameAccumulator = 0.0;
  private sampleHz = 50.0;
  private selectedMotionName: string | null = 'default';
  private currentDone = true;
  private referenceVisible = true;
  refJointPos: Float32Array[] = [];
  refRootPos: Float32Array[] = [];
  refRootQuat: Float32Array[] = [];
  refIdx = 0;
  refLen = 0;
  nJoints = 0;

  constructor(
    _termName: string,
    config: CommandConfigEntry,
    context: CommandTermContext,
  ) {
    this.context = context;
    this.motions = Array.isArray(config.motions) ? config.motions as GentleHumanoidMotionConfig[] : [];
    this.targetJointNames = Array.isArray(config.joint_names)
      ? config.joint_names.filter((value): value is string => typeof value === 'string')
      : [];
    this.nJoints = this.targetJointNames.length;
    const futureSteps = Array.isArray(config.future_steps)
      ? config.future_steps.map((value: number) => Math.floor(value))
      : [0];
    this.futureHistoryLen = Math.max(0, -Math.min(...futureSteps));
    this.switchTailKeepSteps = Math.max(
      this.futureHistoryLen,
      Math.floor(typeof config.switch_tail_keep_steps === 'number' ? config.switch_tail_keep_steps : 8),
    );
    this.transitionSteps = Math.max(0, Math.floor(typeof config.transition_steps === 'number' ? config.transition_steps : 100));
    this.refMaxLen = Math.max(0, Math.floor(typeof config.ref_max_len === 'number' ? config.ref_max_len : 2048));
    const modelJointNames = context.mjModel ? this.getJointNames(context.mjModel) : null;
    this.qposAdr = this.resolveQposAdr(this.targetJointNames, modelJointNames);
    this.qvelAdr = this.resolveQvelAdr(this.targetJointNames, modelJointNames);
    const rootJointIndex = this.findFreeJointIndex();
    this.rootQposAdr = rootJointIndex >= 0 && context.mjModel ? context.mjModel.jnt_qposadr[rootJointIndex] : 0;
    this.rootQvelAdr = rootJointIndex >= 0 && context.mjModel ? context.mjModel.jnt_dofadr[rootJointIndex] : 0;
    this.ghostBodies = new Map();
    this.ghostData = context.mjModel ? new context.mujoco.MjData(context.mjModel) : null;
    this.ghostRoot = this.createGhostRoot();
    this.selectedMotionName =
      this.motions.find((motion) => motion.default)?.name ??
      this.motions.find((motion) => motion.name === 'default')?.name ??
      this.motions[0]?.name ??
      null;
  }

  getCommand(): Float32Array {
    const current = this.refJointPos[this.refIdx] ?? new Float32Array(this.nJoints);
    return new Float32Array(current);
  }

  getUiConfig(): CommandUiConfig | null {
    return null;
  }

  async setSelectedMotion(name: string | null): Promise<boolean> {
    if (name === null) {
      this.selectedMotionName = null;
      this.refJointPos = [];
      this.refRootPos = [];
      this.refRootQuat = [];
      this.refIdx = 0;
      this.refLen = 0;
      this.frameAccumulator = 0.0;
      this.currentDone = true;
      this.updateGhostPose();
      return false;
    }

    const config = this.motions.find((motion) => motion.name === name);
    if (!config) {
      return false;
    }

    const loaded = this.loadedMotions.get(name) ?? await this.loadMotion(config);
    this.loadedMotions.set(name, loaded);
    this.refJointPos = loaded.jointPos;
    this.refRootPos = loaded.rootPos;
    this.refRootQuat = loaded.rootQuat;
    this.refIdx = 0;
    this.refLen = loaded.frameCount;
    this.frameAccumulator = 0.0;
    this.sampleHz = loaded.fps;
    this.selectedMotionName = name;
    this.currentDone = this.refIdx >= this.refLen - 1;
    this.applyReferenceStateToSim();
    this.updateGhostPose();
    return true;
  }

  getSelectedMotionName(): string | null {
    return this.selectedMotionName;
  }

  setReferenceVisible(visible: boolean): void {
    this.referenceVisible = visible;
    this.updateGhostPose();
  }

  reset(): void {
    this.refIdx = 0;
    this.frameAccumulator = 0.0;
    this.currentDone = this.refIdx >= this.refLen - 1;
    this.applyReferenceStateToSim();
    this.updateGhostPose();
  }

  update(dt: number): void {
    if (this.refLen <= 1 || this.currentDone) {
      return;
    }
    this.frameAccumulator += dt * this.sampleHz;
    while (this.frameAccumulator >= 1.0 && !this.currentDone) {
      this.refIdx = Math.min(this.refIdx + 1, this.refLen - 1);
      this.currentDone = this.refIdx >= this.refLen - 1;
      this.frameAccumulator -= 1.0;
    }
    this.updateGhostPose();
  }

  updateDebugVisuals(): void {
    if (this.ghostRoot) {
      this.ghostRoot.visible = this.referenceVisible && this.refLen > 0;
    }
  }

  dispose(): void {
    if (this.ghostRoot) {
      this.ghostRoot.parent?.remove(this.ghostRoot);
      this.ghostRoot.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            for (const material of obj.material) {
              material.dispose?.();
            }
          } else {
            obj.material?.dispose?.();
          }
        }
      });
    }
    this.ghostData?.delete?.();
  }

  isReady(): boolean {
    return this.refLen > 0;
  }

  private createGhostRoot(): THREE.Group | null {
    const bodies = this.context.bodies ?? null;
    const mjModel = this.context.mjModel;
    if (!bodies || !mjModel) {
      return null;
    }

    const root = new THREE.Group();
    root.name = 'GentleHumanoid Tracking Ghost';
    root.visible = false;
    for (const [bodyId, body] of Object.entries(bodies)) {
      const numericBodyId = Number(bodyId);
      if (!this.isDynamicBody(numericBodyId)) {
        continue;
      }

      const clone = body.clone(true);
      clone.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            obj.material = obj.material.map(setGhostMaterial);
          } else {
            obj.material = setGhostMaterial(obj.material);
          }
          obj.renderOrder = 2;
        }
      });
      if (!hasRenderableMesh(clone)) {
        continue;
      }
      this.ghostBodies.set(numericBodyId, clone);
      root.add(clone);
    }

    (this.context.mujocoRoot ?? this.context.scene).add(root);
    return root;
  }

  private isDynamicBody(bodyId: number): boolean {
    const mjModel = this.context.mjModel;
    if (!mjModel || bodyId <= 0 || bodyId >= mjModel.nbody) {
      return false;
    }

    let current = bodyId;
    while (current > 0) {
      if (mjModel.body_jntnum[current] > 0) {
        return true;
      }
      current = mjModel.body_parentid[current];
    }
    return false;
  }

  private async loadMotion(config: GentleHumanoidMotionConfig): Promise<MotionFrames> {
    const npz = await loadNpz(await resolveBytes(config.data));
    const dofPos = npz.dof_pos;
    const rootPos = npz.root_pos;
    const rootRot = npz.root_rot;
    if (!dofPos || !rootPos || !rootRot) {
      throw new Error(`GentleHumanoid motion "${config.name}" must contain dof_pos, root_pos and root_rot`);
    }
    const start = config.metadata?.start ?? 0;
    const end = config.metadata?.end ?? -1;
    const sourceJointNames = npz.joint_names?.strings ?? config.dataset_joint_names ?? this.targetJointNames;
    const jointPos = remapJointFrames(
      sliceFrames(splitFrames(dofPos), start, end),
      sourceJointNames,
      this.targetJointNames,
    );
    const rootPosFrames = sliceFrames(splitFrames(rootPos), start, end);
    const rootQuat = sliceFrames(splitFrames(rootRot), start, end).map((xyzw) =>
      normalizeQuat([xyzw[3] ?? 1.0, xyzw[0] ?? 0.0, xyzw[1] ?? 0.0, xyzw[2] ?? 0.0])
    );
    const frameCount = Math.min(jointPos.length, rootPosFrames.length, rootQuat.length);
    return {
      jointPos: jointPos.slice(0, frameCount),
      rootPos: rootPosFrames.slice(0, frameCount),
      rootQuat: rootQuat.slice(0, frameCount),
      fps: config.fps ?? readScalar(npz.fps, 50.0),
      frameCount,
    };
  }

  private buildAppendSegment(motion: MotionFrames): MotionFrames {
    if (motion.frameCount <= 0) {
      return motion;
    }
    const anchor = this.readRefTailState();
    const aligned = this.alignMotionToAnchor(motion, anchor);
    const transition = this.buildTransitionPrefix(anchor, {
      jointPos: aligned.jointPos[0],
      rootPos: aligned.rootPos[0],
      rootQuat: aligned.rootQuat[0],
    });
    return {
      jointPos: [...transition.jointPos, ...aligned.jointPos],
      rootPos: [...transition.rootPos, ...aligned.rootPos],
      rootQuat: [...transition.rootQuat, ...aligned.rootQuat],
      fps: motion.fps,
      frameCount: transition.frameCount + aligned.frameCount,
    };
  }

  private alignMotionToAnchor(motion: MotionFrames, anchor: RefState): MotionFrames {
    const firstPos = motion.rootPos[0];
    const firstYaw = yawQuat(motion.rootQuat[0]);
    const anchorYaw = yawQuat(anchor.rootQuat);
    const delta = quatMultiply(anchorYaw, quatInverse(firstYaw));
    const jointPos = motion.jointPos.map((frame) => new Float32Array(frame));
    const rootPos = motion.rootPos.map((pos) => {
      const rel = new Float32Array([
        (pos[0] ?? 0.0) - (firstPos[0] ?? 0.0),
        (pos[1] ?? 0.0) - (firstPos[1] ?? 0.0),
        (pos[2] ?? 0.0) - (firstPos[2] ?? 0.0),
      ]);
      const rotated = quatApply(delta, rel);
      return new Float32Array([
        rotated[0] + (anchor.rootPos[0] ?? 0.0),
        rotated[1] + (anchor.rootPos[1] ?? 0.0),
        pos[2] ?? 0.0,
      ]);
    });
    const rootQuat = motion.rootQuat.map((quat) => normalizeQuat(quatMultiply(delta, quat)));
    return { jointPos, rootPos, rootQuat, fps: motion.fps, frameCount: motion.frameCount };
  }

  private buildTransitionPrefix(anchor: RefState, target: RefState): MotionFrames {
    const steps = this.transitionSteps;
    const jointPos: Float32Array[] = [];
    const rootPos: Float32Array[] = [];
    const rootQuat: Float32Array[] = [];
    for (let i = 0; i < steps; i++) {
      const t = (i + 1) / (steps + 1);
      const joints = new Float32Array(this.nJoints);
      for (let j = 0; j < this.nJoints; j++) {
        joints[j] = (anchor.jointPos[j] ?? 0.0) * (1.0 - t) + (target.jointPos[j] ?? 0.0) * t;
      }
      jointPos.push(joints);
      rootPos.push(new Float32Array([
        (anchor.rootPos[0] ?? 0.0) * (1.0 - t) + (target.rootPos[0] ?? 0.0) * t,
        (anchor.rootPos[1] ?? 0.0) * (1.0 - t) + (target.rootPos[1] ?? 0.0) * t,
        (anchor.rootPos[2] ?? 0.0) * (1.0 - t) + (target.rootPos[2] ?? 0.0) * t,
      ]));
      rootQuat.push(slerpQuat(anchor.rootQuat, target.rootQuat, t));
    }
    return { jointPos, rootPos, rootQuat, fps: this.sampleHz, frameCount: steps };
  }

  private appendRefFrames(frames: MotionFrames): void {
    if (frames.frameCount <= 0) {
      return;
    }
    this.refJointPos.push(...frames.jointPos.map((frame) => new Float32Array(frame)));
    this.refRootPos.push(...frames.rootPos.map((frame) => new Float32Array(frame)));
    this.refRootQuat.push(...frames.rootQuat.map((frame) => normalizeQuat(frame)));
    this.refLen = this.refJointPos.length;
    this.currentDone = this.refIdx >= this.refLen - 1;
    this.trimRefPrefix();
  }

  private readRefTailState(): RefState {
    if (this.refLen > 0) {
      const last = this.refLen - 1;
      return {
        jointPos: new Float32Array(this.refJointPos[last]),
        rootPos: new Float32Array(this.refRootPos[last]),
        rootQuat: new Float32Array(this.refRootQuat[last]),
      };
    }
    return this.readCurrentState();
  }

  private readCurrentState(): RefState {
    const mjData = this.context.mjData;
    const jointPos = new Float32Array(this.nJoints);
    if (mjData) {
      for (let i = 0; i < this.qposAdr.length; i++) {
        jointPos[i] = mjData.qpos[this.qposAdr[i]] ?? 0.0;
      }
    }
    const rootPos = mjData
      ? new Float32Array([
        mjData.qpos[this.rootQposAdr + 0] ?? 0.0,
        mjData.qpos[this.rootQposAdr + 1] ?? 0.0,
        mjData.qpos[this.rootQposAdr + 2] ?? 0.78,
      ])
      : new Float32Array([0.0, 0.0, 0.78]);
    const rootQuat = mjData
      ? normalizeQuat([
        mjData.qpos[this.rootQposAdr + 3] ?? 1.0,
        mjData.qpos[this.rootQposAdr + 4] ?? 0.0,
        mjData.qpos[this.rootQposAdr + 5] ?? 0.0,
        mjData.qpos[this.rootQposAdr + 6] ?? 0.0,
      ])
      : new Float32Array([1.0, 0.0, 0.0, 0.0]);
    return { jointPos, rootPos, rootQuat };
  }

  private trimRefPrefix(): void {
    const keepHist = Math.max(this.futureHistoryLen, this.switchTailKeepSteps) + 2;
    let drop = Math.max(0, this.refIdx - keepHist);
    if (this.refMaxLen > 0) {
      const overflow = Math.max(0, this.refLen - this.refMaxLen);
      drop = Math.max(drop, Math.min(overflow, Math.max(0, this.refIdx - keepHist)));
    }
    if (drop <= 0) {
      return;
    }
    this.refJointPos = this.refJointPos.slice(drop);
    this.refRootPos = this.refRootPos.slice(drop);
    this.refRootQuat = this.refRootQuat.slice(drop);
    this.refIdx -= drop;
    this.refLen = this.refJointPos.length;
    this.currentDone = this.refIdx >= this.refLen - 1;
  }

  private applyReferenceStateToSim(): void {
    const mjModel = this.context.mjModel;
    const mjData = this.context.mjData;
    if (!mjModel || !mjData || !this.writeReferenceStateToData(mjData)) {
      return;
    }
    this.context.mujoco.mj_forward(mjModel, mjData);
  }

  private writeReferenceStateToData(data: import('mujoco').MjData): boolean {
    if (this.refLen <= 0) {
      return false;
    }

    const rootPos = this.refRootPos[this.refIdx];
    const rootQuat = this.refRootQuat[this.refIdx];
    const jointPos = this.refJointPos[this.refIdx];
    if (!rootPos || !rootQuat || !jointPos) {
      return false;
    }

    data.qpos[this.rootQposAdr + 0] = rootPos[0] ?? 0.0;
    data.qpos[this.rootQposAdr + 1] = rootPos[1] ?? 0.0;
    data.qpos[this.rootQposAdr + 2] = rootPos[2] ?? 0.78;
    data.qpos[this.rootQposAdr + 3] = rootQuat[0] ?? 1.0;
    data.qpos[this.rootQposAdr + 4] = rootQuat[1] ?? 0.0;
    data.qpos[this.rootQposAdr + 5] = rootQuat[2] ?? 0.0;
    data.qpos[this.rootQposAdr + 6] = rootQuat[3] ?? 0.0;
    for (let i = 0; i < this.qposAdr.length; i++) {
      data.qpos[this.qposAdr[i]] = jointPos[i] ?? 0.0;
    }

    for (let i = 0; i < 6; i++) {
      data.qvel[this.rootQvelAdr + i] = 0.0;
    }
    for (const adr of this.qvelAdr) {
      data.qvel[adr] = 0.0;
    }
    return true;
  }

  private updateGhostPose(): void {
    const mjModel = this.context.mjModel;
    if (!this.ghostRoot || !this.ghostData || !mjModel || this.refLen <= 0) {
      if (this.ghostRoot) {
        this.ghostRoot.visible = false;
      }
      return;
    }

    this.ghostData.qpos.set(mjModel.qpos0);
    this.ghostData.qvel.fill(0.0);
    if (!this.writeReferenceStateToData(this.ghostData)) {
      this.ghostRoot.visible = false;
      return;
    }

    this.context.mujoco.mj_forward(mjModel, this.ghostData);
    for (const [bodyId, body] of this.ghostBodies) {
      getPosition(this.ghostData.xpos, bodyId, body.position);
      getQuaternion(this.ghostData.xquat, bodyId, body.quaternion);
    }
    this.ghostRoot.visible = this.referenceVisible;
  }

  private findFreeJointIndex(): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_type[i] === 0) {
        return i;
      }
    }
    return -1;
  }

  private resolveQposAdr(jointNames: string[], availableJointNames: string[] | null): number[] {
    const mjModel = this.context.mjModel;
    if (!mjModel || !availableJointNames) {
      return Array.from({ length: jointNames.length }, () => 0);
    }
    return jointNames.map((name) => {
      const idx = availableJointNames.indexOf(name);
      if (idx < 0) {
        throw new Error(`GentleHumanoidTrackingCommand: joint "${name}" not found in model`);
      }
      return mjModel.jnt_qposadr[idx];
    });
  }

  private resolveQvelAdr(jointNames: string[], availableJointNames: string[] | null): number[] {
    const mjModel = this.context.mjModel;
    if (!mjModel || !availableJointNames) {
      return Array.from({ length: jointNames.length }, () => 0);
    }
    return jointNames.map((name) => {
      const idx = availableJointNames.indexOf(name);
      if (idx < 0) {
        throw new Error(`GentleHumanoidTrackingCommand: joint "${name}" not found in model`);
      }
      return mjModel.jnt_dofadr[idx];
    });
  }

  private getJointNames(mjModel: import('mujoco').MjModel): string[] {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const names: string[] = [];
    for (let j = 0; j < mjModel.njnt; j++) {
      let start = mjModel.name_jntadr[j];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      let name = decoder.decode(namesArray.subarray(start, end));
      if (!name && j === 0) {
        name = 'floating_base_joint';
      }
      names.push(name);
    }
    return names;
  }
}
