import type { ObservationConstructor } from '../policy/PolicyRunner';
import { CustomObservations } from './custom_observations';

// All built-in observations are declarative composition graphs now (computed by
// DslObservation via the engine primitive registry; see ADR 0003), so there are
// no named built-in observation classes.  This registry only carries ts_src
// custom observations (resolved by name via the legacy path, e.g. the Mimic /
// gentle-humanoid demos).

export const Observations: Record<string, ObservationConstructor> =
  CustomObservations as unknown as Record<string, ObservationConstructor>;
