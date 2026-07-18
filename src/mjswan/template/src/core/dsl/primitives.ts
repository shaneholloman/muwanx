/**
 * Engine primitive op registry — the closed set of operators DSL composition
 * graphs may reference (see ADR 0003).
 *
 * Each primitive takes the resolved input values plus the node's `attrs` and
 * the `EvalContext` (carrying the current PolicyRunner / state / params /
 * mjData reference) and returns a single value.  Adding an op here is the
 * only path for an engine PR to grow the declarative surface.
 */

import type { PolicyState } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { DslPrimitiveValue } from './types';
import { isTrackingSource } from '../command';
import {
  quatApplyInv,
  quatInverse,
  quatMultiply,
  quatToRot6d,
  quatToRot6dColumns,
} from '../observation/math';

export type EvalContext = {
  runner: PolicyRunner;
  state: PolicyState;
  params: Record<string, unknown>;
};

export type NodeState = Record<string, unknown>;

export type Primitive = (
  inputs: DslPrimitiveValue[],
  attrs: Record<string, unknown>,
  ctx: EvalContext,
  state: NodeState,
) => DslPrimitiveValue;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function asFloat(value: DslPrimitiveValue, index = 0): number {
  if (typeof value === 'number') return value;
  if (value instanceof Float32Array) return value[index] ?? 0;
  if (typeof value === 'boolean') return value ? 1 : 0;
  return 0;
}

function asVec(value: DslPrimitiveValue): Float32Array {
  if (value instanceof Float32Array) return value;
  if (typeof value === 'number') return new Float32Array([value]);
  return new Float32Array(0);
}

function bodyIdByName(runner: PolicyRunner, bodyName: string): number {
  const mjModel = runner.getContext()?.mjModel ?? null;
  if (!mjModel) return -1;
  for (let i = 0; i < mjModel.nbody; i++) {
    const name = mjModel.body(i).name;
    if (name === bodyName || name.endsWith(`/${bodyName}`)) return i;
  }
  return -1;
}

type TrackingSource = {
  isReady(): boolean;
  getAnchorPos(): Float32Array | null;
  getAnchorQuat(): Float32Array | null;
  getBodyPosW(): Float32Array | null;
  getBodyNames(): string[];
  getAnchorBodyName(): string | null;
};

/** The active motion tracking command, or null when none is ready. */
function trackingTerm(ctx: EvalContext): TrackingSource | null {
  const term = ctx.runner.getContext()?.commandManager?.getTerm('motion');
  return isTrackingSource(term) && term.isReady() ? (term as TrackingSource) : null;
}

/**
 * A motion command exposing reference-trajectory buffers (root pose + per-joint
 * targets) at a play cursor.  This is a distinct capability from the anchor-
 * based {@link TrackingSource}; both the built-in `TrackingCommand` and bespoke
 * motion players satisfy it by declaring these public fields.
 */
type MotionRefSource = {
  isReady(): boolean;
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  refJointPos: Float32Array[];
  refIdx: number;
  refLen: number;
};

/** The active motion command carrying reference buffers, or null. */
function motionRefTerm(ctx: EvalContext): MotionRefSource | null {
  const term = ctx.runner.getContext()?.commandManager?.getTerm('motion');
  if (
    typeof term === 'object' &&
    term !== null &&
    'refRootPos' in term &&
    'refRootQuat' in term &&
    'refJointPos' in term &&
    typeof (term as { isReady?: unknown }).isReady === 'function'
  ) {
    return term as unknown as MotionRefSource;
  }
  return null;
}

/** Add `step` to `base`, clamped to `[0, len-1]` (matches clampFutureIndices). */
function clampRefIdx(base: number, step: number, len: number): number {
  const i = base + step;
  if (i < 0) return 0;
  if (i >= len) return Math.max(0, len - 1);
  return i;
}

// --- Model name-table decoders + address resolution --------------------------

function decodeNames(
  mjModel: import('mujoco').MjModel,
  count: number,
  adr: ArrayLike<number>,
): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let i = 0; i < count; i++) {
    let start = adr[i];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

function modelJointNames(mjModel: import('mujoco').MjModel): string[] {
  return decodeNames(mjModel, mjModel.njnt, mjModel.name_jntadr);
}

/** Resolve joint names to qpos addresses; unknown names map to -1. */
function jointQposAdr(runner: PolicyRunner, jointNames: string[]): number[] {
  const mjModel = runner.getContext()?.mjModel ?? null;
  if (!mjModel) return jointNames.map(() => -1);
  const names = modelJointNames(mjModel);
  return jointNames.map((jn) => {
    const idx = names.indexOf(jn);
    return idx < 0 ? -1 : mjModel.jnt_qposadr[idx];
  });
}

/** Resolve joint names to qvel (dof) addresses; unknown names map to -1. */
function jointDofAdr(runner: PolicyRunner, jointNames: string[]): number[] {
  const mjModel = runner.getContext()?.mjModel ?? null;
  if (!mjModel) return jointNames.map(() => -1);
  const names = modelJointNames(mjModel);
  return jointNames.map((jn) => {
    const idx = names.indexOf(jn);
    return idx < 0 ? -1 : mjModel.jnt_dofadr[idx];
  });
}

function siteIdByName(runner: PolicyRunner, siteName: string): number {
  const mjModel = runner.getContext()?.mjModel ?? null;
  if (!mjModel) return -1;
  const names = decodeNames(mjModel, mjModel.nsite, mjModel.name_siteadr);
  return names.indexOf(siteName);
}

function sensorAdrDim(runner: PolicyRunner, sensorName: string): { adr: number; dim: number } {
  const mjModel = runner.getContext()?.mjModel ?? null;
  if (!mjModel) return { adr: 0, dim: 0 };
  const names = decodeNames(mjModel, mjModel.nsensor, mjModel.name_sensoradr);
  const idx = names.indexOf(sensorName);
  if (idx < 0) return { adr: 0, dim: 0 };
  return { adr: mjModel.sensor_adr[idx], dim: mjModel.sensor_dim[idx] };
}

function strList(value: unknown): string[] | null {
  return Array.isArray(value) && value.every((v) => typeof v === 'string')
    ? (value as string[])
    : null;
}

// ---------------------------------------------------------------------------
// Primitive registry
// ---------------------------------------------------------------------------

export const Primitives: Record<string, Primitive> = {
  // -- Constants & lookups ------------------------------------------------
  Const: (_in, attrs) => Number(attrs.value ?? 0),

  // -- Stateful: episode step counter -------------------------------------
  StepCount: (_in, _attrs, _ctx, state) => {
    const next = ((state.count as number | undefined) ?? 0) + 1;
    state.count = next;
    return next;
  },

  Param: (_in, attrs, ctx) => {
    const raw = ctx.params[String(attrs.name)];
    if (typeof raw === 'number') return raw;
    if (Array.isArray(raw)) return Float32Array.from(raw as number[]);
    return 0;
  },

  // -- Entity-bound sources (alias the existing PolicyState slots) --------
  RootAngVelB: (_in, _attrs, ctx) => {
    return ctx.state.rootAngVel ? new Float32Array(ctx.state.rootAngVel) : new Float32Array(3);
  },
  RootLinkLinVelB: (_in, _attrs, ctx) => {
    return ctx.state.rootLinVel ? new Float32Array(ctx.state.rootLinVel) : new Float32Array(3);
  },
  RootLinkPosW: (_in, _attrs, ctx) => {
    return ctx.state.rootPos ? new Float32Array(ctx.state.rootPos) : new Float32Array(3);
  },
  RootLinkQuatW: (_in, _attrs, ctx) => {
    return ctx.state.rootQuat ? new Float32Array(ctx.state.rootQuat) : new Float32Array([1, 0, 0, 0]);
  },

  // Gravity vector (world [0, 0, -1]) projected into the entity base frame.
  // Derived from rootQuat; matches mjlab's `asset.data.projected_gravity_b`.
  ProjectedGravityB: (_in, _attrs, ctx) => {
    const q = ctx.state.rootQuat ?? new Float32Array([1, 0, 0, 0]);
    // Inverse-rotate (0, 0, -1) by q.  Using mjlab's quat-apply-inv formula
    // (q is (w, x, y, z) in MuJoCo convention):
    //   t = 2 * cross(q.xyz, v)
    //   v' = v - w * t + cross(q.xyz, t)
    const w = q[0], x = q[1], y = q[2], z = q[3];
    const vx = 0, vy = 0, vz = -1;
    const tx = 2.0 * (y * vz - z * vy);
    const ty = 2.0 * (z * vx - x * vz);
    const tz = 2.0 * (x * vy - y * vx);
    const cx = y * tz - z * ty;
    const cy = z * tx - x * tz;
    const cz = x * ty - y * tx;
    return new Float32Array([vx - w * tx + cx, vy - w * ty + cy, vz - w * tz + cz]);
  },

  // -- Elementwise arithmetic ---------------------------------------------
  Add: (inputs) => elementwise(inputs[0], inputs[1], (a, b) => a + b),
  Sub: (inputs) => elementwise(inputs[0], inputs[1], (a, b) => a - b),
  Mul: (inputs) => elementwise(inputs[0], inputs[1], (a, b) => a * b),
  Neg: (inputs) => unary(inputs[0], (a) => -a),
  Abs: (inputs) => unary(inputs[0], Math.abs),
  Acos: (inputs) => unary(inputs[0], (a) => Math.acos(Math.max(-1, Math.min(1, a)))),

  // -- Indexing ------------------------------------------------------------
  Index: (inputs, attrs) => {
    const vec = asVec(inputs[0]);
    const idx = Number(attrs.i ?? 0);
    return vec[idx] ?? 0;
  },

  // -- Comparisons (produce Uint8Array for vectors, boolean for scalars) --
  Gt: (inputs) => compare(inputs[0], inputs[1], (a, b) => a > b),
  Lt: (inputs) => compare(inputs[0], inputs[1], (a, b) => a < b),
  Ge: (inputs) => compare(inputs[0], inputs[1], (a, b) => a >= b),
  Le: (inputs) => compare(inputs[0], inputs[1], (a, b) => a <= b),

  // -- Boolean reductions --------------------------------------------------
  Any: (inputs) => reduceBool(inputs[0], (acc, v) => acc || v, false),
  All: (inputs) => reduceBool(inputs[0], (acc, v) => acc && v, true),

  // -- Boolean elementwise -------------------------------------------------
  Or: (inputs) => boolElementwise(inputs[0], inputs[1], (a, b) => a || b),
  And: (inputs) => boolElementwise(inputs[0], inputs[1], (a, b) => a && b),
  Not: (inputs) => {
    const x = inputs[0];
    if (typeof x === 'boolean') return !x;
    if (typeof x === 'number') return x === 0;
    if (x instanceof Uint8Array) {
      const out = new Uint8Array(x.length);
      for (let i = 0; i < x.length; i++) out[i] = x[i] ? 0 : 1;
      return out;
    }
    return false;
  },

  // -- Constant vector literal --------------------------------------------
  ConstVec: (_in, attrs) => {
    const vals = Array.isArray(attrs.values) ? (attrs.values as number[]) : [];
    return Float32Array.from(vals);
  },

  // -- Stateful: capture input on first eval after reset, then hold -------
  SpawnCapture: (inputs, _attrs, _ctx, state) => {
    if (state.captured === undefined) {
      state.captured = asVec(inputs[0]).slice();
    }
    return state.captured as Float32Array;
  },

  // -- Quaternion inverse-apply (rotate vec by q^-1) ----------------------
  QuatApplyInv: (inputs) => Float32Array.from(quatApplyInv(asVec(inputs[0]), asVec(inputs[1]))),

  // -- Motion-tracking sources --------------------------------------------
  TrackingAnchorPos: (_in, _attrs, ctx) => {
    const pos = trackingTerm(ctx)?.getAnchorPos() ?? null;
    return pos ? new Float32Array(pos) : new Float32Array(3);
  },
  TrackingAnchorQuat: (_in, _attrs, ctx) => {
    const quat = trackingTerm(ctx)?.getAnchorQuat() ?? null;
    return quat ? new Float32Array(quat) : new Float32Array([1, 0, 0, 0]);
  },
  TrackingCurrentAnchorPos: (_in, _attrs, ctx) => {
    const term = trackingTerm(ctx);
    const name = term?.getAnchorBodyName() ?? null;
    const mjData = ctx.runner.getContext()?.mjData ?? null;
    if (!name || !mjData) return new Float32Array(3);
    const id = bodyIdByName(ctx.runner, name);
    if (id < 0) return new Float32Array(3);
    return new Float32Array([mjData.xpos[id * 3], mjData.xpos[id * 3 + 1], mjData.xpos[id * 3 + 2]]);
  },
  TrackingCurrentAnchorQuat: (_in, _attrs, ctx) => {
    const term = trackingTerm(ctx);
    const name = term?.getAnchorBodyName() ?? null;
    const mjData = ctx.runner.getContext()?.mjData ?? null;
    if (!name || !mjData) return new Float32Array([1, 0, 0, 0]);
    const id = bodyIdByName(ctx.runner, name);
    if (id < 0) return new Float32Array([1, 0, 0, 0]);
    return new Float32Array(mjData.xquat.slice(id * 4, id * 4 + 4));
  },
  TrackingBodyPosZDeviationMax: (_in, attrs, ctx) => {
    const term = trackingTerm(ctx);
    const mjData = ctx.runner.getContext()?.mjData ?? null;
    if (!term || !mjData) return 0;
    const refBodyPosW = term.getBodyPosW();
    if (!refBodyPosW) return 0;
    const allBodies = term.getBodyNames();
    const wanted = Array.isArray(attrs.body_names) && attrs.body_names.length > 0
      ? (attrs.body_names as string[])
      : allBodies;
    let maxDev = 0;
    for (const name of wanted) {
      const slot = allBodies.indexOf(name);
      const id = bodyIdByName(ctx.runner, name);
      if (slot < 0 || id < 0) continue;
      const refZ = refBodyPosW[slot * 3 + 2] ?? 0;
      const curZ = mjData.xpos[id * 3 + 2] ?? 0;
      const dev = Math.abs(refZ - curZ);
      if (dev > maxDev) maxDev = dev;
    }
    return maxDev;
  },

  // -- Observation post-processing (baked from scale/clip/history_length) --
  Clip: (inputs, attrs) => {
    const lo = Number(attrs.min ?? -Infinity);
    const hi = Number(attrs.max ?? Infinity);
    return unary(inputs[0], (a) => Math.min(hi, Math.max(lo, a)));
  },
  History: (inputs, attrs, _ctx, state) => {
    const steps = Math.max(1, Math.floor(Number(attrs.steps ?? 1)));
    const cur = asVec(inputs[0]);
    let buf = state.buffer as Float32Array[] | undefined;
    if (buf === undefined || buf.length !== steps || buf[0].length !== cur.length) {
      buf = Array.from({ length: steps }, () => cur.slice());
      state.buffer = buf;
    } else {
      for (let i = buf.length - 1; i > 0; i--) buf[i].set(buf[i - 1]);
      buf[0].set(cur);
    }
    if (steps === 1) return new Float32Array(buf[0]);
    const len = cur.length;
    const out = new Float32Array(len * steps);
    if (attrs.interleaved) {
      // Joint-major (transpose): [v0_t, v0_{t-1}, …, v1_t, …] — Isaac convention.
      for (let j = 0; j < len; j++) {
        for (let i = 0; i < steps; i++) out[j * steps + i] = buf[i][j];
      }
    } else {
      // Step-major: [all_t, all_{t-1}, …].
      for (let i = 0; i < steps; i++) out.set(buf[i], i * len);
    }
    return out;
  },
  Concat: (inputs) => {
    let total = 0;
    const vecs = inputs.map((v) => asVec(v));
    for (const v of vecs) total += v.length;
    const out = new Float32Array(total);
    let offset = 0;
    for (const v of vecs) {
      out.set(v, offset);
      offset += v.length;
    }
    return out;
  },
  Cos: (inputs) => unary(inputs[0], Math.cos),
  Sin: (inputs) => unary(inputs[0], Math.sin),

  // -- Quaternion algebra --------------------------------------------------
  QuatMul: (inputs) => Float32Array.from(quatMultiply(asVec(inputs[0]), asVec(inputs[1]))),
  QuatInv: (inputs) => Float32Array.from(quatInverse(asVec(inputs[0]))),
  QuatToRot6d: (inputs) => Float32Array.from(quatToRot6d(asVec(inputs[0]))),

  // -- Joint state ---------------------------------------------------------
  // joint_names: a list resolves explicit qpos addresses; "all"/absent reads
  // the policy joint-position vector (matches the legacy JointPos class).
  JointPos: (_in, attrs, ctx) => {
    const names = strList(attrs.joint_names);
    if (names === null) {
      const jp = ctx.state.jointPos;
      return jp ? new Float32Array(jp) : new Float32Array(ctx.runner.getNumActions());
    }
    const adrs = jointQposAdr(ctx.runner, names);
    const qpos = ctx.runner.getContext()?.mjData?.qpos;
    const out = new Float32Array(adrs.length);
    for (let i = 0; i < adrs.length; i++) {
      out[i] = adrs[i] >= 0 && qpos !== undefined ? qpos[adrs[i]] : 0;
    }
    return out;
  },
  DefaultJointPos: (_in, attrs, ctx) => {
    const names = strList(attrs.joint_names);
    if (names === null) {
      return new Float32Array(ctx.runner.getDefaultJointPos());
    }
    const adrs = jointQposAdr(ctx.runner, names);
    const qpos0 = ctx.runner.getContext()?.mjModel?.qpos0;
    const out = new Float32Array(adrs.length);
    for (let i = 0; i < adrs.length; i++) {
      out[i] = adrs[i] >= 0 && qpos0 !== undefined ? qpos0[adrs[i]] : 0;
    }
    return out;
  },
  JointVel: (_in, attrs, ctx) => {
    const names = strList(attrs.joint_names);
    if (names === null) {
      const jv = ctx.state.jointVel;
      return jv ? new Float32Array(jv) : new Float32Array(ctx.runner.getNumActions());
    }
    const adrs = jointDofAdr(ctx.runner, names);
    const qvel = ctx.runner.getContext()?.mjData?.qvel;
    const out = new Float32Array(adrs.length);
    for (let i = 0; i < adrs.length; i++) {
      out[i] = adrs[i] >= 0 && qvel !== undefined ? qvel[adrs[i]] : 0;
    }
    return out;
  },

  // -- Actions (current policy output; wrap with History for a stack) ------
  PrevAction: (_in, _attrs, ctx) => new Float32Array(ctx.runner.getLastActions()),

  // -- Commands ------------------------------------------------------------
  CommandValue: (_in, attrs, ctx) => {
    const name = String(attrs.command ?? '');
    const cmd = name ? ctx.runner.getContext()?.commandManager?.getCommand(name) : null;
    return cmd ? new Float32Array(cmd) : new Float32Array(0);
  },

  // -- Sensors -------------------------------------------------------------
  Sensor: (_in, attrs, ctx) => {
    const { adr, dim } = sensorAdrDim(ctx.runner, String(attrs.sensor ?? ''));
    const data = ctx.runner.getContext()?.mjData?.sensordata;
    const out = new Float32Array(dim);
    if (data !== undefined) {
      for (let i = 0; i < dim; i++) out[i] = data[adr + i] ?? 0;
    }
    return out;
  },

  // -- World-frame poses by name (current robot state) ---------------------
  SitePos: (_in, attrs, ctx) => {
    const id = siteIdByName(ctx.runner, String(attrs.site ?? ''));
    const sx = ctx.runner.getContext()?.mjData?.site_xpos;
    if (id < 0 || sx === undefined) return new Float32Array(3);
    return new Float32Array([sx[id * 3], sx[id * 3 + 1], sx[id * 3 + 2]]);
  },
  BodyPos: (_in, attrs, ctx) => {
    const id = bodyIdByName(ctx.runner, String(attrs.body ?? ''));
    const xpos = ctx.runner.getContext()?.mjData?.xpos;
    if (id < 0 || xpos === undefined) return new Float32Array(3);
    return new Float32Array([xpos[id * 3], xpos[id * 3 + 1], xpos[id * 3 + 2]]);
  },
  BodyQuat: (_in, attrs, ctx) => {
    const id = bodyIdByName(ctx.runner, String(attrs.body ?? ''));
    const xquat = ctx.runner.getContext()?.mjData?.xquat;
    if (id < 0 || xquat === undefined) return new Float32Array([1, 0, 0, 0]);
    return new Float32Array(xquat.slice(id * 4, id * 4 + 4));
  },

  // -- Motion reference trajectory, sampled at a lookahead offset ----------
  // `field`: 'root_pos' (3) | 'root_quat' (4) | 'joint_pos' (nJoints).
  // `step`: offset added to the current refIdx, clamped to [0, refLen-1].
  // When no motion is ready, returns a finite fallback (identity quat / zeros)
  // so downstream quaternion math stays defined; wrap the term with
  // `Mul(..., TrackingIsReady)` to zero it out (matches the bespoke
  // `if (!ready) return zeros`).
  TrackingRefField: (_in, attrs, ctx) => {
    const field = String(attrs.field ?? '');
    const fallback = () =>
      field === 'root_quat'
        ? new Float32Array([1, 0, 0, 0])
        : field === 'root_pos'
          ? new Float32Array(3)
          : new Float32Array(ctx.runner.getNumActions());
    const term = motionRefTerm(ctx);
    if (!term || !term.isReady()) return fallback();
    const buf =
      field === 'root_pos'
        ? term.refRootPos
        : field === 'root_quat'
          ? term.refRootQuat
          : term.refJointPos;
    const frame = buf[clampRefIdx(term.refIdx, Number(attrs.step ?? 0), term.refLen)];
    return frame ? new Float32Array(frame) : fallback();
  },

  // 1.0 when a motion reference is loaded and ready, else 0.0.
  TrackingIsReady: (_in, _attrs, ctx) => {
    const term = motionRefTerm(ctx);
    return term && term.isReady() ? 1.0 : 0.0;
  },

  // Column-major rot6d (first two rotation columns; see quatToRot6dColumns).
  QuatToRot6dColumns: (inputs) =>
    Float32Array.from(quatToRot6dColumns(asVec(inputs[0]))),

  // L2-normalize; a zero vector maps to itself (norm floored to 1), matching
  // the bespoke `Math.hypot(...) || 1.0`.
  Normalize: (inputs) => {
    const v = asVec(inputs[0]);
    const n = Math.hypot(...v) || 1.0;
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
    return out;
  },

  // -- Tracking reference, per body (world frame) --------------------------
  TrackingRefBodyPos: (_in, attrs, ctx) => {
    const term = trackingTerm(ctx);
    if (!term) return new Float32Array(3);
    const refBodyPosW = term.getBodyPosW();
    if (!refBodyPosW) return new Float32Array(3);
    const slot = term.getBodyNames().indexOf(String(attrs.body ?? ''));
    if (slot < 0) return new Float32Array(3);
    return new Float32Array([
      refBodyPosW[slot * 3],
      refBodyPosW[slot * 3 + 1],
      refBodyPosW[slot * 3 + 2],
    ]);
  },
};

function boolElementwise(
  a: DslPrimitiveValue,
  b: DslPrimitiveValue,
  op: (x: boolean, y: boolean) => boolean,
): DslPrimitiveValue {
  const aBool = (v: DslPrimitiveValue, i: number): boolean => {
    if (typeof v === 'boolean') return v;
    if (typeof v === 'number') return v !== 0;
    if (v instanceof Uint8Array) return (v[i] ?? 0) !== 0;
    if (v instanceof Float32Array) return (v[i] ?? 0) !== 0;
    return false;
  };
  const aIsVec = a instanceof Uint8Array || a instanceof Float32Array;
  const bIsVec = b instanceof Uint8Array || b instanceof Float32Array;
  if (!aIsVec && !bIsVec) return op(aBool(a, 0), aBool(b, 0));
  const aLen = aIsVec ? (a as Uint8Array | Float32Array).length : 1;
  const bLen = bIsVec ? (b as Uint8Array | Float32Array).length : 1;
  const len = Math.max(aLen, bLen);
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = op(aBool(a, i), aBool(b, i)) ? 1 : 0;
  return out;
}

// ---------------------------------------------------------------------------
// Helper implementations
// ---------------------------------------------------------------------------

function unary(input: DslPrimitiveValue, op: (a: number) => number): DslPrimitiveValue {
  if (typeof input === 'number') return op(input);
  if (input instanceof Float32Array) {
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) out[i] = op(input[i]);
    return out;
  }
  return op(asFloat(input));
}

function elementwise(
  a: DslPrimitiveValue,
  b: DslPrimitiveValue,
  op: (x: number, y: number) => number,
): DslPrimitiveValue {
  const aIsVec = a instanceof Float32Array;
  const bIsVec = b instanceof Float32Array;
  if (!aIsVec && !bIsVec) return op(asFloat(a), asFloat(b));
  const len = Math.max(aIsVec ? a.length : 1, bIsVec ? (b as Float32Array).length : 1);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const av = aIsVec ? a[i] ?? 0 : asFloat(a);
    const bv = bIsVec ? (b as Float32Array)[i] ?? 0 : asFloat(b);
    out[i] = op(av, bv);
  }
  return out;
}

function compare(
  a: DslPrimitiveValue,
  b: DslPrimitiveValue,
  op: (x: number, y: number) => boolean,
): DslPrimitiveValue {
  const aIsVec = a instanceof Float32Array;
  const bIsVec = b instanceof Float32Array;
  if (!aIsVec && !bIsVec) return op(asFloat(a), asFloat(b));
  const len = Math.max(aIsVec ? a.length : 1, bIsVec ? (b as Float32Array).length : 1);
  const out = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    const av = aIsVec ? a[i] ?? 0 : asFloat(a);
    const bv = bIsVec ? (b as Float32Array)[i] ?? 0 : asFloat(b);
    out[i] = op(av, bv) ? 1 : 0;
  }
  return out;
}

function reduceBool(
  input: DslPrimitiveValue,
  op: (acc: boolean, v: boolean) => boolean,
  init: boolean,
): DslPrimitiveValue {
  if (typeof input === 'boolean') return input;
  if (typeof input === 'number') return input !== 0;
  let acc = init;
  if (input instanceof Uint8Array) {
    for (let i = 0; i < input.length; i++) acc = op(acc, input[i] !== 0);
    return acc;
  }
  if (input instanceof Float32Array) {
    for (let i = 0; i < input.length; i++) acc = op(acc, input[i] !== 0);
    return acc;
  }
  return acc;
}

// Re-export helper so primitives implemented in other files can share it.
export { bodyIdByName };
