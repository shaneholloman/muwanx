import { CustomEvents } from './custom_events';

export type { EventConstructor } from './EventBase';

// All built-in reset events are declarative mutation envelopes now (applied by
// DslEvent; see ADR 0003), so there are no named built-in event classes.  This
// registry only carries ts_src custom events (resolved by name).

export const Events: Record<string, import('./EventBase').EventConstructor> = {
  ...CustomEvents,
};
