/**
 * mjlab's `time_out`, native because its body compares env-level step counters rather
 * than reading entity state — there is nothing to put in a graph. Evaluates the build's
 * `elapsed_s >= episode_length_s` against time the manager accumulates from `dt`.
 *
 * A *truncation*, not a failure: the manager keeps the two apart.
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
    // No finite horizon never times out; nor does a missing value, which would fire always.
    const declared = config.episode_length_s ?? 0;
    this.episodeLengthS = declared > 0 ? declared : Number.POSITIVE_INFINITY;
    this.getElapsedS = getElapsedS;
  }

  evaluate(_state: PolicyState): boolean {
    return this.getElapsedS() >= this.episodeLengthS;
  }
}
