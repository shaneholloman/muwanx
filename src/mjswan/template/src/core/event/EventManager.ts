import type { SeededRng } from '../rng';
import type { OnnxSessionCache } from '../onnx/session';
import type { SlotReader } from '../onnx/session';
import { EventBase, type EventConfig, type EventContext } from './EventBase';
import type { EventConstructor } from './EventBase';
import { DslEvent } from './DslEvent';
import { OnnxEvent, isOnnxEventConfig } from './OnnxEvent';
import { IntervalTrigger, ResetTrigger, StartupTrigger } from './triggers';

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

/** Deps for constructing ONNX-backed event terms; omit if a scene has none. */
export interface EventManagerDeps {
  sessions: OnnxSessionCache;
  rng: SeededRng;
  readSlot?: SlotReader;
}

type LegacyTerm = { kind: 'legacy'; term: EventBase; trigger: ResetTrigger };
type OnnxResetTerm = { kind: 'onnx'; term: OnnxEvent; trigger: ResetTrigger };
type ResetEntry = LegacyTerm | OnnxResetTerm;

/**
 * Native Event dispatch (ADR 0005 §5, companion brief §4).
 *
 * Previously reset-only (`onReset()`); now mode-aware. `mode="reset"` terms
 * (legacy `DslEvent`/registry classes, or `OnnxEvent`) fire on episode reset,
 * gated by a `ResetTrigger`; `mode="interval"` terms fire on the frames their
 * `IntervalTrigger` allows; `mode="startup"` terms fire once. Fusion (brief §4)
 * only changes how many graphs exist — dispatch here still gates *every* call,
 * fused or not, so a quiet frame costs no `ort.run()`.
 */
export class EventManager {
  private resetTerms: ResetEntry[] = [];
  private intervalTerms: Array<{ term: OnnxEvent; trigger: IntervalTrigger }> = [];
  private startupTerms: Array<{ term: OnnxEvent; trigger: StartupTrigger }> = [];

  constructor(
    configs: EventConfig[],
    registry: Record<string, EventConstructor>,
    deps?: EventManagerDeps,
  ) {
    for (const config of configs) {
      if (isOnnxEventConfig(config)) {
        if (!deps) {
          console.warn(
            `[EventManager] ONNX event "${config.name}" needs session/rng deps; none supplied.`,
          );
          continue;
        }
        const session = deps.sessions.get(config.onnx);
        if (!session) {
          console.warn(`[EventManager] No ONNX session loaded for "${config.onnx}".`);
          continue;
        }
        const term = new OnnxEvent(config, { session, rng: deps.rng, readSlot: deps.readSlot });
        switch (config.mode) {
          case 'startup':
            this.startupTerms.push({ term, trigger: new StartupTrigger() });
            break;
          case 'interval':
            this.intervalTerms.push({
              term,
              trigger: new IntervalTrigger(
                {
                  intervalRangeS: config.interval_range_s ?? [1.0, 1.0],
                  isGlobalTime: config.is_global_time,
                },
                deps.rng,
              ),
            });
            break;
          case 'reset':
          default:
            this.resetTerms.push({
              kind: 'onnx',
              term,
              trigger: new ResetTrigger({
                minStepCountBetweenReset: config.min_step_count_between_reset,
              }),
            });
            break;
        }
        continue;
      }
      if (isDslEvent(config)) {
        this.resetTerms.push({ kind: 'legacy', term: new DslEvent(config), trigger: new ResetTrigger() });
        continue;
      }
      const EventClass = registry[config.name];
      if (!EventClass) {
        console.warn(`[EventManager] Unknown event type: ${config.name}`);
        continue;
      }
      this.resetTerms.push({ kind: 'legacy', term: new EventClass(config), trigger: new ResetTrigger() });
    }
  }

  /** Fire every `mode="startup"` term once. Call after the scene/policy loads. */
  async startup(context: EventContext): Promise<void> {
    await Promise.all(
      this.startupTerms.filter(({ trigger }) => trigger.take()).map(({ term }) => term.fire(context)),
    );
  }

  /**
   * Advance one control step: fires `mode="interval"` terms whose timer has
   * elapsed (fire-and-forget; `OnnxEvent` itself skips a term already in
   * flight) and advances every reset-gate step counter.
   */
  tick(dt: number, context: EventContext): void {
    for (const { term, trigger } of this.intervalTerms) {
      if (trigger.tick(dt)) void term.fire(context);
    }
    for (const { trigger } of this.resetTerms) trigger.step();
  }

  /** Fire every `mode="reset"` term whose gate allows it; awaited by the caller. */
  async onReset(context: EventContext): Promise<void> {
    await Promise.all(
      this.resetTerms.map(async (entry) => {
        if (!entry.trigger.take()) return;
        if (entry.kind === 'onnx') await entry.term.fire(context);
        else entry.term.onReset(context);
      }),
    );
    for (const { trigger } of this.intervalTerms) trigger.onReset();
  }

  get size(): number {
    return this.resetTerms.length + this.intervalTerms.length + this.startupTerms.length;
  }
}
