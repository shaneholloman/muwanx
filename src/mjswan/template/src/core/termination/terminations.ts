import { TerminationBase, type TerminationConfig } from './TerminationBase';

export type TerminationConstructor = new (
  runner: import('../policy/PolicyRunner').PolicyRunner,
  config: TerminationConfig,
) => TerminationBase;
