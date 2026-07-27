/**
 * `TimeOutTermination`: mjlab's `time_out`, kept native (ADR 0005 §2).
 *
 * mjlab's own body compares env-level step counters rather than reading any
 * entity state, so the tracer classifies it as native by construction — there is
 * nothing to put in a graph. The build emits
 * `{native: "elapsed_s >= episode_length_s", episode_length_s}` and this term
 * evaluates exactly that comparison against elapsed wall-clock episode time the
 * manager accumulates from the control `dt`.
 *
 * Note this is a *truncation*, not a failure: the manager keeps `time_out` terms
 * separate so a timeout reports `truncated` while a real termination reports
 * `terminated`.
 */

import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

export interface TimeOutTerminationConfig extends TerminationConfig {
  episode_length_s?: number;
}

export class TimeOutTermination extends TerminationBase {
  private readonly episodeLengthS: number;
  private getElapsedS: () => number;

  constructor(
    runner: PolicyRunner,
    config: TimeOutTerminationConfig,
    getElapsedS: () => number,
  ) {
    super(runner, config);
    // A task with no finite horizon (mjlab's play configs set an effectively
    // infinite episode_length_s) never times out; treat a missing or
    // non-positive value the same way rather than firing every frame.
    const declared = config.episode_length_s ?? 0;
    this.episodeLengthS = declared > 0 ? declared : Number.POSITIVE_INFINITY;
    this.getElapsedS = getElapsedS;
  }

  evaluate(_state: PolicyState): boolean {
    return this.getElapsedS() >= this.episodeLengthS;
  }
}
