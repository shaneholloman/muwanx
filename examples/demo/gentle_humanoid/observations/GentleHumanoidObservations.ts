import { ObservationBase } from 'mjswan/observation';
import type { ObservationConfig } from 'mjswan/observation';
import { normalizeQuat, quatInverse, quatMultiply, quatApplyInv, clampFutureIndices } from 'mjswan/math';
import type { PolicyState } from 'mjswan/types';
import type { PolicyRunner } from 'mjswan/types';

type GentleHumanoidTrackingSource = {
  refJointPos: Float32Array[];
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  refIdx: number;
  refLen: number;
  nJoints: number;
  isReady(): boolean;
};

function getTracking(runner: PolicyRunner): GentleHumanoidTrackingSource | null {
  const term = runner.getContext()?.commandManager?.getTerm('motion');
  if (
    typeof term === 'object' &&
    term !== null &&
    'refJointPos' in term &&
    'refRootPos' in term &&
    'refRootQuat' in term &&
    typeof (term as unknown as GentleHumanoidTrackingSource).isReady === 'function'
  ) {
    return term as unknown as GentleHumanoidTrackingSource;
  }
  return null;
}

function readSteps(config: ObservationConfig, key: string, fallback: number[]): number[] {
  const values = config[key];
  if (!Array.isArray(values) || values.length === 0) {
    return fallback;
  }
  return values.map((value: number) => Math.floor(value));
}


function quatToRot6dColumns(q: ArrayLike<number>): number[] {
  const [w, x, y, z] = normalizeQuat(q);
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  const r00 = 1.0 - 2.0 * (yy + zz);
  const r01 = 2.0 * (xy - wz);
  const r10 = 2.0 * (xy + wz);
  const r11 = 1.0 - 2.0 * (xx + zz);
  const r20 = 2.0 * (xz - wy);
  const r21 = 2.0 * (yz + wx);
  return [r00, r10, r20, r01, r11, r21];
}

function readCompliance(runner: PolicyRunner, config: ObservationConfig): { enabled: number; force: number } {
  const command = runner.getContext()?.commandManager?.getCommand((config.command_name as string | undefined) ?? 'compliance') ?? new Float32Array(0);
  const enabled = command.length > 0 ? (command[0] >= 0.5 ? 1.0 : 0.0) : Number(config.default_enabled ?? 1.0);
  const force = command.length > 1 ? command[1] : Number(config.default_force ?? 10.0);
  return { enabled, force };
}

function normalizeGravity(q: ArrayLike<number>): Float32Array {
  const g = quatApplyInv(q, [0.0, 0.0, -1.0]);
  const n = Math.hypot(g[0], g[1], g[2]) || 1.0;
  return new Float32Array([g[0] / n, g[1] / n, g[2] / n]);
}

abstract class HistoryObservation extends ObservationBase {
  protected readonly steps: number[];
  private readonly width: number;
  private readonly history: Float32Array[];

  constructor(runner: PolicyRunner, config: ObservationConfig, stepKey: string, fallback: number[], width: number) {
    super(runner, config);
    this.steps = readSteps(config, stepKey, fallback);
    this.width = width;
    const maxStep = Math.max(0, ...this.steps);
    this.history = Array.from({ length: maxStep + 1 }, () => new Float32Array(width));
  }

  get size(): number {
    return this.steps.length * this.width;
  }

  reset(state?: PolicyState): void {
    const current = this.computeCurrent(state);
    for (const frame of this.history) {
      frame.set(current);
    }
  }

  update(state: PolicyState): void {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.computeCurrent(state));
  }

  compute(): Float32Array {
    const out = new Float32Array(this.size);
    let offset = 0;
    for (const step of this.steps) {
      out.set(this.history[Math.min(step, this.history.length - 1)], offset);
      offset += this.width;
    }
    return out;
  }

  protected abstract computeCurrent(state?: PolicyState): Float32Array;
}

export class GentleHumanoidBootIndicator extends ObservationBase {
  get size(): number {
    return 1;
  }

  compute(): Float32Array {
    // The browser replay has no deployment boot phase, so keep this policy flag disabled.
    return new Float32Array([0.0]);
  }
}

export class GentleHumanoidTrackingCommandObsRaw extends ObservationBase {
  private readonly futureSteps: number[];
  private readonly outputLength: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.futureSteps = readSteps(config, 'future_steps', [0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]);
    const nFut = this.futureSteps.length;
    this.outputLength = (nFut - 1) * 3 + nFut * 6;
  }

  get size(): number {
    return this.outputLength;
  }

  compute(state: PolicyState): Float32Array {
    const tracking = getTracking(this.runner);
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const basePos = tracking.refRootPos[indices[0]];
    const baseQuat = tracking.refRootQuat[indices[0]];
    const out: number[] = [];
    for (let i = 1; i < indices.length; i++) {
      const pos = tracking.refRootPos[indices[i]];
      const diffB = quatApplyInv(baseQuat, [
        (pos[0] ?? 0.0) - (basePos[0] ?? 0.0),
        (pos[1] ?? 0.0) - (basePos[1] ?? 0.0),
        (pos[2] ?? 0.0) - (basePos[2] ?? 0.0),
      ]);
      out.push(diffB[0], diffB[1], diffB[2]);
    }
    const qCurInv = quatInverse(state.rootQuat ?? [1.0, 0.0, 0.0, 0.0]);
    for (const idx of indices) {
      const rel = quatMultiply(qCurInv, tracking.refRootQuat[idx]);
      out.push(...quatToRot6dColumns(rel));
    }
    return Float32Array.from(out);
  }
}

export class GentleHumanoidComplianceFlagObs extends ObservationBase {
  get size(): number {
    return 3;
  }

  compute(): Float32Array {
    const { enabled, force } = readCompliance(this.runner, this.config);
    return new Float32Array([enabled, enabled * force, enabled * force / 0.05]);
  }
}

export class GentleHumanoidTargetJointPosObs extends ObservationBase {
  private readonly futureSteps: number[];
  private readonly nJoints: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.futureSteps = readSteps(config, 'future_steps', [0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]);
    this.nJoints = typeof config.num_joints === 'number' ? Math.floor(config.num_joints) : runner.getNumActions();
  }

  get size(): number {
    return this.futureSteps.length * this.nJoints * 2;
  }

  compute(state: PolicyState): Float32Array {
    const tracking = getTracking(this.runner);
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const out = new Float32Array(this.size);
    let targetOffset = 0;
    let diffOffset = indices.length * this.nJoints;
    const current = state.jointPos ?? new Float32Array(this.nJoints);
    for (const idx of indices) {
      const target = tracking.refJointPos[idx];
      for (let j = 0; j < this.nJoints; j++) {
        const value = target[j] ?? 0.0;
        out[targetOffset++] = value;
        out[diffOffset++] = value - (current[j] ?? 0.0);
      }
    }
    return out;
  }
}

export class GentleHumanoidTargetRootZObs extends ObservationBase {
  private readonly futureSteps: number[];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.futureSteps = readSteps(config, 'future_steps', [0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]);
  }

  get size(): number {
    return this.futureSteps.length;
  }

  compute(): Float32Array {
    const tracking = getTracking(this.runner);
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const out = new Float32Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
      out[i] = tracking.refRootPos[indices[i]][2] ?? 0.0;
    }
    return out;
  }
}

export class GentleHumanoidTargetProjectedGravityBObs extends ObservationBase {
  private readonly futureSteps: number[];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.futureSteps = readSteps(config, 'future_steps', [0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]);
  }

  get size(): number {
    return this.futureSteps.length * 3;
  }

  compute(): Float32Array {
    const tracking = getTracking(this.runner);
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const out = new Float32Array(this.size);
    let offset = 0;
    for (const idx of indices) {
      out.set(normalizeGravity(tracking.refRootQuat[idx]), offset);
      offset += 3;
    }
    return out;
  }
}

export class GentleHumanoidRootAngVelBHistory extends HistoryObservation {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 'history_steps', [0, 1, 2, 3, 4, 8, 12, 16, 20], 3);
  }

  protected computeCurrent(state?: PolicyState): Float32Array {
    return new Float32Array(state?.rootAngVel ?? new Float32Array(3));
  }
}

export class GentleHumanoidProjectedGravityBHistory extends HistoryObservation {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 'history_steps', [0, 1, 2, 3, 4, 8, 12, 16, 20], 3);
  }

  protected computeCurrent(state?: PolicyState): Float32Array {
    return normalizeGravity(state?.rootQuat ?? [1.0, 0.0, 0.0, 0.0]);
  }
}

export class GentleHumanoidJointPosHistory extends HistoryObservation {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    const width = typeof config.num_joints === 'number' ? Math.floor(config.num_joints) : runner.getNumActions();
    super(runner, config, 'history_steps', [0, 1, 2, 3, 4, 8, 12, 16, 20], width);
  }

  protected computeCurrent(state?: PolicyState): Float32Array {
    return new Float32Array(state?.jointPos ?? new Float32Array(this.runner.getNumActions()));
  }
}

export class GentleHumanoidJointVelHistory extends HistoryObservation {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    const width = typeof config.num_joints === 'number' ? Math.floor(config.num_joints) : runner.getNumActions();
    super(runner, config, 'history_steps', [0, 1, 2, 3, 4, 8, 12, 16, 20], width);
  }

  protected computeCurrent(state?: PolicyState): Float32Array {
    return new Float32Array(state?.jointVel ?? new Float32Array(this.runner.getNumActions()));
  }
}

export class GentleHumanoidPrevActions extends ObservationBase {
  private readonly steps: number;
  private readonly numActions: number;
  private readonly history: Float32Array[];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.steps = Math.max(1, Math.floor(typeof config.history_steps === 'number' ? config.history_steps : 8));
    this.numActions = runner.getNumActions();
    this.history = Array.from({ length: this.steps }, () => new Float32Array(this.numActions));
  }

  get size(): number {
    return this.steps * this.numActions;
  }

  reset(): void {
    for (const frame of this.history) {
      frame.fill(0.0);
    }
  }

  update(): void {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.runner.getLastActions());
  }

  compute(): Float32Array {
    const out = new Float32Array(this.size);
    let offset = 0;
    for (const frame of this.history) {
      out.set(frame, offset);
      offset += this.numActions;
    }
    return out;
  }
}
