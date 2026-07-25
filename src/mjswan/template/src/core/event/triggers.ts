/**
 * Native Event-mode dispatch (ADR 0005 §5, companion brief §4).
 *
 * Ports the *semantics* of mjlab's `EventManager.apply()` — interval countdown,
 * interval-time resampling, startup-once, and reset gating — as plain scalars.
 * mjlab's `(num_envs,)` tensors, `.nonzero()` and `env_ids` masks exist to make
 * thousands of environments efficient in one GPU kernel launch during training;
 * at N=1 browser playback none of that payoff exists, so `time_left` is a
 * number and `fired` is a boolean.
 *
 * These triggers decide *when* a term runs. They never run it — the caller
 * invokes the term's ONNX graph (or native handler) only on the frames a trigger
 * fires, so a gated term costs no `ort.run()` on quiet frames (brief §4: fusion
 * reduces graph count, never call frequency).
 */

import type { SeededRng } from '../rng';

export type EventMode = 'startup' | 'reset' | 'interval';

export interface IntervalTriggerConfig {
  /** `[min, max]` seconds between firings, resampled after each firing. */
  intervalRangeS: readonly [number, number];
  /**
   * mjlab's `is_global_time`. At N=1 the distinction is only whether the timer
   * survives an episode reset (`true`) or restarts with the episode (`false`).
   */
  isGlobalTime?: boolean;
}

/** Countdown timer for one `mode="interval"` event term. */
export class IntervalTrigger {
  private timeLeft: number;

  constructor(
    private readonly config: IntervalTriggerConfig,
    private readonly rng: SeededRng,
  ) {
    this.timeLeft = this.sampleInterval();
  }

  /**
   * Advance by `dt` seconds; returns true on the frames the term should run.
   *
   * Mirrors mjlab: decrement, fire when the timer reaches zero, then resample
   * the next interval. Carries the overshoot so a long `dt` cannot drift the
   * average rate; a `dt` spanning several intervals still fires once (playback
   * has no use for catch-up bursts of the same disturbance).
   */
  tick(dt: number): boolean {
    this.timeLeft -= dt;
    if (this.timeLeft > 0) return false;
    this.timeLeft += this.sampleInterval();
    if (this.timeLeft <= 0) this.timeLeft = this.sampleInterval();
    return true;
  }

  /** Episode reset: per-episode timers restart, global timers keep running. */
  onReset(): void {
    if (!this.config.isGlobalTime) this.timeLeft = this.sampleInterval();
  }

  get secondsUntilNextFiring(): number {
    return this.timeLeft;
  }

  private sampleInterval(): number {
    const [min, max] = this.config.intervalRangeS;
    return this.rng.uniform(min, max);
  }
}

/** Fires exactly once, at engine init — mjlab's `mode="startup"`. */
export class StartupTrigger {
  private fired = false;

  /** True only the first time it is called; false forever after. */
  take(): boolean {
    if (this.fired) return false;
    this.fired = true;
    return true;
  }

  get hasFired(): boolean {
    return this.fired;
  }
}

export interface ResetTriggerConfig {
  /**
   * mjlab's `min_step_count_between_reset`: suppress a reset-mode term when
   * episodes reset in quick succession, so a disturbance term cannot fire every
   * frame during a divergence loop.
   */
  minStepCountBetweenReset?: number;
}

/** Gate for one `mode="reset"` event term. */
export class ResetTrigger {
  private stepsSinceFired = Number.POSITIVE_INFINITY;

  constructor(private readonly config: ResetTriggerConfig = {}) {}

  /** Count a physics/control step toward the gate. */
  step(): void {
    if (this.stepsSinceFired !== Number.POSITIVE_INFINITY) this.stepsSinceFired++;
  }

  /** Called on episode reset; true if the term should run this reset. */
  take(): boolean {
    const min = this.config.minStepCountBetweenReset ?? 0;
    if (this.stepsSinceFired < min) return false;
    this.stepsSinceFired = 0;
    return true;
  }
}
