import { TerminationBase, type TerminationConfig } from './TerminationBase';
import { CustomTerminations } from './custom_terminations';

// Every built-in termination is a traced ONNX graph or the native `time_out`
// marker (ADR 0005), so there are no named built-in termination classes. This
// registry only carries `ts_src` custom terminations, resolved by name.

export type TerminationConstructor = new (
  runner: import('../policy/PolicyRunner').PolicyRunner,
  config: TerminationConfig,
) => TerminationBase;

export const Terminations: Record<string, TerminationConstructor> = {
  ...CustomTerminations,
};
