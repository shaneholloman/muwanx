import type { PolicyState } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';

export type ObservationConfig = {
  name: string;
  [key: string]: unknown;
};

export abstract class ObservationBase<TConfig extends ObservationConfig = ObservationConfig> {
  protected runner: PolicyRunner;
  protected config: TConfig;

  constructor(runner: PolicyRunner, config: TConfig) {
    this.runner = runner;
    this.config = config;
  }

  abstract get size(): number;

  reset?(_state?: PolicyState): void;

  update?(_state: PolicyState): void;

  /** Return a promise that resolves once the observation is ready to compute. */
  preload?(): Promise<void>;

  /**
   * Produce this frame's value.
   *
   * May return a promise: an ONNX-backed term (ADR 0005) runs ORT inference,
   * which is async, and observations are awaited rather than skipped because they
   * feed the policy directly (§8). The group awaits all terms in parallel, so a
   * synchronous term costs nothing extra.
   */
  abstract compute(state: PolicyState): Float32Array | number[] | Promise<Float32Array>;
}
