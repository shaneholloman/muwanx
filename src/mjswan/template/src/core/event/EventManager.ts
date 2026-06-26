import { EventBase, type EventConfig, type EventContext } from './EventBase';
import type { EventConstructor } from './EventBase';
import { DslEvent } from './DslEvent';

function isDslEvent(config: EventConfig): config is EventConfig & {
  kind: 'event';
  mutations: unknown[];
} {
  return (
    'kind' in config
    && (config as { kind?: unknown }).kind === 'event'
    && Array.isArray((config as { mutations?: unknown }).mutations)
  );
}

export class EventManager {
  private terms: EventBase[] = [];

  constructor(
    configs: EventConfig[],
    registry: Record<string, EventConstructor>
  ) {
    for (const config of configs) {
      if (isDslEvent(config)) {
        this.terms.push(new DslEvent(config));
        continue;
      }
      const EventClass = registry[config.name];
      if (!EventClass) {
        console.warn(`[EventManager] Unknown event type: ${config.name}`);
        continue;
      }
      this.terms.push(new EventClass(config));
    }
  }

  onReset(context: EventContext): void {
    for (const term of this.terms) {
      term.onReset(context);
    }
  }

  get size(): number {
    return this.terms.length;
  }
}
