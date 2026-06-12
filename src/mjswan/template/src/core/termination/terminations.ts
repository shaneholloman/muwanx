import { TerminationBase, type TerminationConfig } from './TerminationBase';
import { CustomTerminations } from './custom_terminations';

// All built-in terminations are declarative composition graphs now (evaluated
// by DslTermination via the engine primitive registry; see ADR 0003), so there
// are no named built-in termination classes.  This registry only carries
// ts_src custom terminations (resolved by name via the legacy path).

export type TerminationConstructor = new (config: TerminationConfig) => TerminationBase;

export const Terminations: Record<string, TerminationConstructor> = {
  ...CustomTerminations,
};
