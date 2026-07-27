import type { ObservationConstructor } from '../policy/PolicyRunner';
import { CustomObservations } from './custom_observations';

// Every built-in observation is a traced ONNX graph or a native marker (ADR 0005),
// both handled by one generic class configured entirely by data, so there are no
// named built-in observation classes. This registry only carries `ts_src` custom
// observations, resolved by name (e.g. the Mimic / gentle-humanoid demos).

export const Observations: Record<string, ObservationConstructor> =
  CustomObservations as unknown as Record<string, ObservationConstructor>;
