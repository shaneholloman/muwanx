import { TerminationBase, type TerminationConfig } from './TerminationBase';
import { CustomTerminations } from './custom_terminations';

// Built-ins are traced graphs or the native `time_out`, so this holds only `ts_src` terms.

export type TerminationConstructor = new (
  runner: import('../policy/PolicyRunner').PolicyRunner,
  config: TerminationConfig,
) => TerminationBase;

export const Terminations: Record<string, TerminationConstructor> = {
  ...CustomTerminations,
};
