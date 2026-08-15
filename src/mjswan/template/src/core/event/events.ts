import { CustomEvents } from './custom_events';

export type { EventConstructor } from './EventBase';

// Every built-in event is a traced graph, so this holds only `ts_src` custom events.

export const Events: Record<string, import('./EventBase').EventConstructor> = {
  ...CustomEvents,
};
