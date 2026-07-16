import type { PolicyState } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';

export type TerminationConfig = {
  name: string;
  params?: Record<string, unknown>;
  time_out?: boolean;
};

export abstract class TerminationBase {
  protected runner: PolicyRunner;
  protected config: TerminationConfig;

  constructor(runner: PolicyRunner, config: TerminationConfig) {
    this.runner = runner;
    this.config = config;
  }

  abstract evaluate(state: PolicyState): boolean;

  reset?(): void;
}
