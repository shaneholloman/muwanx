import { CustomEvents } from './custom_events';

export type { EventConstructor } from './EventBase';

// Every built-in event is a traced ONNX graph (ADR 0005), so there are no named
// built-in event classes. This registry only carries `ts_src` custom events,
// resolved by name.

export const Events: Record<string, import('./EventBase').EventConstructor> = {
  ...CustomEvents,
};
