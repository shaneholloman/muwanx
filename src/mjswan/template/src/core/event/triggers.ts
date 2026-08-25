/**
 * The semantics of mjlab's `EventManager.apply()` — interval countdown and resampling,
 * startup-once, reset gating — as plain scalars: mjlab's `(num_envs,)` tensors buy
 * nothing at the browser's N=1.
 *
 * A trigger decides *when* a term runs, never runs it, so a gated term costs no
 * `ort.run()` on quiet frames.
 */

import type { SeededRng } from '../rng';

export type EventMode = 'startup' | 'reset' | 'interval' | 'manual';

export interface IntervalTriggerConfig {
  /** `[min, max]` seconds between firings, resampled after each firing. */
  intervalRangeS: readonly [number, number];
  /** mjlab's `is_global_time`: whether the timer survives an episode reset. */
  isGlobalTime?: boolean;
}

/** Countdown timer for one `mode="interval"` event term. */
export class IntervalTrigger {
  private timeLeft: number;
  private armed = true;

  constructor(
    private readonly config: IntervalTriggerConfig,
    private readonly rng: SeededRng,
  ) {
    this.timeLeft = this.sampleInterval();
  }

  /**
   * Advance by `dt`; true on the frames the term should run. Carries the overshoot so a
   * long `dt` cannot drift the average rate, but still fires only once.
   */
  tick(dt: number): boolean {
    if (!this.armed) return false;
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

  /**
   * A disarmed timer does not count down, and re-arming samples a fresh interval: a
   * countdown that had run while disarmed would fire the instant it came back.
   */
  setArmed(armed: boolean): void {
    if (armed === this.armed) return;
    this.armed = armed;
    if (armed) this.timeLeft = this.sampleInterval();
  }

  get isArmed(): boolean {
    return this.armed;
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
   * mjlab's `min_step_count_between_reset`: suppress a reset-mode term when episodes reset
   * in quick succession, so a disturbance cannot fire every frame during a divergence loop.
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
