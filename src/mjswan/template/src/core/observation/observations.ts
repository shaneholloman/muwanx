import type { ObservationConstructor } from '../policy/PolicyRunner';
import { CustomObservations } from './custom_observations';

// Built-in observations are traced graphs or native markers, both data-configured, so
// this holds only `ts_src` custom terms.

export const Observations: Record<string, ObservationConstructor> =
  CustomObservations as unknown as Record<string, ObservationConstructor>;
