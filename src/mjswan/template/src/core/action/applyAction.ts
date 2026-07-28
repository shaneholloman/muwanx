/**
 * Native action application (ADR 0005 §7).
 *
 * Action is the one manager with no ONNX at all: a closed built-in set, the
 * hottest loop, run once per physics substep. mjlab models it the same way —
 * `ActionTermCfg` is an ABC with a `build()`, not an author `func` — so the
 * browser mirrors the *kinds* rather than tracing anything.
 *
 * Split out of `runtime.ts` because this is the one part of the MDP step that
 * decides what force reaches the sim, and it was reachable only through a
 * private method on a THREE-and-DOM-bound class. Here it is a free function over
 * `mjData`, so the rollout-parity harness can drive it in Node against the real
 * MuJoCo WASM (ADR §Consequences' "mandatory validation") — the same reason the
 * managers are DOM-free.
 */

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** mjlab's `ActionTermCfg` kinds this runtime implements. */
export type ControlType = 'joint_position' | 'torque' | 'muscle_activation';

/**
 * One resolved action term: the build's descriptor with every name already
 * turned into an address into this model.
 */
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
   * Per-actuator: true = position actuator (`biastype=affine`), so `ctrl` is a
   * target position and MuJoCo runs the PD itself; false = motor actuator
   * (`biastype=none`), so `ctrl` is a torque and the PD is computed here.
   */
  positionActuator: boolean[];
  kp: Float32Array;
  kd: Float32Array;
  /** `muscle_activation` only: MyoSuite sigmoid when true, else clip to [0, 1]. */
  muscleNormalize: boolean;
  /**
   * Per-target bounds on the *processed* action, `±Infinity` where unbounded.
   *
   * mjlab's `BaseAction.process_actions` clamps `raw * scale + offset` and only
   * then does each kind's `apply_actions` run — for `joint_position` that means the
   * clamp happens *before* the encoder-bias subtraction, not on the final target.
   * Clamping the wrong side of that would move every bound by the bias.
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
      // `encoder_bias` is subtracted before the write (ADR §7): the policy was
      // trained against a biased reading, so the target has to be un-biased to
      // land where the policy meant. The clip lands on the processed action that
      // precedes it, which is where mjlab puts it.
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
        // `+ offset` matches mjlab's `raw * scale + offset`; this branch used to
        // drop it, which is invisible while every effort term's offset is 0.0 (as
        // all the reference tasks' are) and silently wrong for one that sets it.
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
      // The declared bounds apply to the processed action, so before the
      // activation mapping — the [0, 1] squeeze below is the actuator's own range.
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
 * Per-target bounds from a pattern-keyed `clip` config.
 *
 * mjlab resolves these with `re.fullmatch` against the term's target names, so the
 * anchored regex here is the same rule — and it is why `clip` cannot reuse the
 * exact-name resolver `stiffness`/`damping` share: those are mjswan's own fields.
 * A target no pattern matches stays unbounded.
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

/**
 * Run one control step's physics: `decimation` substeps, each re-applying the
 * action before integrating.
 *
 * The action is re-applied every substep rather than once per control step
 * because a motor actuator's PD term reads live `qpos`/`qvel` — mjlab's
 * `apply_action()` is inside its decimation loop for the same reason.
 */
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
