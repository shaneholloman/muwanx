/**
 * The one manager with no ONNX: a closed built-in set on the hottest loop, run once per
 * physics substep. mjlab models it the same way (`ActionTermCfg` is an ABC, not an
 * author `func`), so this mirrors the *kinds* rather than tracing anything.
 *
 * A free function over `mjData` rather than a method on the DOM-bound runtime, so the
 * rollout-parity harness can drive it in Node against the real MuJoCo WASM.
 */

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** mjlab's `ActionTermCfg` kinds this runtime implements. */
export type ControlType = 'joint_position' | 'torque' | 'muscle_activation';

/** One resolved action term: the build's descriptor, names already resolved to addresses. */
export interface ResolvedActionTerm {
  controlType: string;
  /** `mjData.ctrl` address per joint; `< 0` for a joint with no actuator. */
  ctrlAdr: number[];
  qposAdr: number[];
  qvelAdr: number[];
  /** Indices into the flat policy action vector for this term's joints. */
  actionIndices: number[];
  actionScale: Float32Array;
  actionOffset: Float32Array;
  defaultJointPos: Float32Array;
  encoderBias: Float32Array;
  /**
   * Per-actuator: true = position (`biastype=affine`), so `ctrl` is a target and MuJoCo
   * runs the PD; false = motor, so `ctrl` is a torque and the PD is computed here.
   */
  positionActuator: boolean[];
  kp: Float32Array;
  kd: Float32Array;
  /** `muscle_activation` only: MyoSuite sigmoid when true, else clip to [0, 1]. */
  muscleNormalize: boolean;
  /**
   * Per-target bounds on the *processed* action (`raw * scale + offset`), `±Infinity`
   * where unbounded. mjlab clamps there, before `joint_position` subtracts the encoder
   * bias — clamping the other side would move every bound by the bias.
   */
  clipLo: Float32Array;
  clipHi: Float32Array;
}

/**
 * Write `mjData.ctrl` for one control step from the policy's action vector.
 *
 * Zeroes `ctrl` first, so an actuator no term claims stays at rest rather than
 * holding the previous step's value.
 */
export function applyAction(
  mjData: MjData,
  terms: readonly ResolvedActionTerm[],
  actions: Float32Array,
): void {
  const ctrl = mjData.ctrl;
  ctrl.fill(0.0);
  for (const term of terms) applyActionTerm(mjData, term, actions);
}

function applyActionTerm(
  mjData: MjData,
  term: ResolvedActionTerm,
  actions: Float32Array,
): void {
  const {
    controlType,
    ctrlAdr,
    qposAdr,
    qvelAdr,
    actionIndices,
    actionScale,
    actionOffset,
    defaultJointPos,
    encoderBias,
    positionActuator,
    kp,
    kd,
    muscleNormalize,
  } = term;
  const numJoints = ctrlAdr.length;
  const ctrl = mjData.ctrl;

  if (controlType === 'joint_position') {
    for (let i = 0; i < numJoints; i++) {
      const ctrlIndex = ctrlAdr[i];
      if (ctrlIndex < 0) continue;
      const actionValue = actions[actionIndices[i]] ?? 0;
      // Un-bias the target: the policy was trained against a biased reading.
      const processed = clamp(
        defaultJointPos[i] + actionOffset[i] + actionScale[i] * actionValue,
        term.clipLo[i],
        term.clipHi[i],
      );
      const target = processed - encoderBias[i];

      if (positionActuator[i]) {
        ctrl[ctrlIndex] = target;
      } else {
        const qpos = mjData.qpos[qposAdr[i]];
        const qvel = mjData.qvel[qvelAdr[i]];
        ctrl[ctrlIndex] = kp[i] * (target - qpos) + kd[i] * (0 - qvel);
      }
    }
    return;
  }

  if (controlType === 'torque') {
    for (let i = 0; i < numJoints; i++) {
      const ctrlIndex = ctrlAdr[i];
      if (ctrlIndex >= 0) {
        // `+ offset`, as mjlab does — 0.0 on every reference task, so easily missed.
        ctrl[ctrlIndex] = clamp(
          actionScale[i] * (actions[actionIndices[i]] ?? 0) + actionOffset[i],
          term.clipLo[i],
          term.clipHi[i],
        );
      }
    }
    return;
  }

  if (controlType === 'muscle_activation') {
    // Shared pre-step: raw = scale * action + offset.
    // normalize=true:  MyoSuite-canonical sigmoid σ(5 * (raw - 0.5)).
    // normalize=false: clip(raw, 0, 1) for models that already output excitation.
    for (let i = 0; i < numJoints; i++) {
      const ctrlIndex = ctrlAdr[i];
      if (ctrlIndex < 0) continue;
      // The bounds apply before the activation mapping; [0, 1] is the actuator's range.
      const raw = clamp(
        (actions[actionIndices[i]] ?? 0) * actionScale[i] + actionOffset[i],
        term.clipLo[i],
        term.clipHi[i],
      );
      ctrl[ctrlIndex] = muscleNormalize
        ? 1 / (1 + Math.exp(-5 * (raw - 0.5)))
        : clamp01(raw);
    }
  }
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

function clamp(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value;
}

/**
 * Per-target bounds from a pattern-keyed `clip` config. Anchored, as mjlab's
 * `re.fullmatch` is — which is why this cannot reuse the exact-name resolver
 * `stiffness`/`damping` share. An unmatched target stays unbounded.
 */
export function resolveActionClip(
  clip: Record<string, readonly number[]> | undefined,
  targetNames: readonly string[],
  length: number,
): { clipLo: Float32Array; clipHi: Float32Array } {
  const clipLo = new Float32Array(length).fill(-Infinity);
  const clipHi = new Float32Array(length).fill(Infinity);
  if (!clip) return { clipLo, clipHi };
  for (const [pattern, bounds] of Object.entries(clip)) {
    if (!Array.isArray(bounds) || bounds.length < 2) {
      console.warn(`[applyAction] clip "${pattern}" needs [min, max]; ignoring.`);
      continue;
    }
    let matched = 0;
    const re = new RegExp(`^(?:${pattern})$`);
    for (let i = 0; i < Math.min(length, targetNames.length); i++) {
      if (!re.test(targetNames[i])) continue;
      clipLo[i] = bounds[0];
      clipHi[i] = bounds[1];
      matched++;
    }
    if (matched === 0) {
      console.warn(`[applyAction] clip "${pattern}" matched no target; ignoring.`);
    }
  }
  return { clipLo, clipHi };
}

/** `decimation` substeps, re-applying the action each time: a motor's PD reads live state. */
export function stepPhysics(
  mujoco: { mj_step(model: MjModel, data: MjData): void },
  mjModel: MjModel,
  mjData: MjData,
  terms: readonly ResolvedActionTerm[],
  actions: Float32Array,
  decimation: number,
  onSubstep?: () => void,
): void {
  for (let substep = 0; substep < decimation; substep++) {
    applyAction(mjData, terms, actions);
    onSubstep?.();
    mujoco.mj_step(mjModel, mjData);
  }
}
