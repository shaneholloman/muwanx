import type { SeededRng } from '../rng';
import type { OnnxSessionCache } from '../onnx/session';
import type { SlotReader } from '../onnx/session';
import { EventBase, type EventConfig, type EventContext } from './EventBase';
import type { EventConstructor } from './EventBase';
import {
  applyModelFieldDr,
  isModelFieldDrConfig,
  ModelFieldDefaults,
  type ModelFieldDrConfig,
} from './modelFieldDr';
import { OnnxEvent, isOnnxEventConfig } from './OnnxEvent';
import { IntervalTrigger, ResetTrigger, StartupTrigger } from './triggers';

/** Deps for constructing ONNX-backed event terms; omit if a scene has none. */
export interface EventManagerDeps {
  sessions: OnnxSessionCache;
  rng: SeededRng;
  readSlot?: SlotReader;
}

/** A plugin-supplied event class (ADR 0004 §10) — reset-only, no traced graph. */
type PluginTerm = { kind: 'plugin'; term: EventBase; trigger: ResetTrigger };
type OnnxResetTerm = { kind: 'onnx'; term: OnnxEvent; trigger: ResetTrigger };
type ResetEntry = PluginTerm | OnnxResetTerm;

/**
 * Native Event dispatch (ADR 0005 §5, companion brief §4).
 *
 * Previously reset-only (`onReset()`); now mode-aware. `mode="reset"` terms
 * (a traced `OnnxEvent`, or a plugin-registered class) fire on episode reset,
 * gated by a `ResetTrigger`; `mode="interval"` terms fire on the frames their
 * `IntervalTrigger` allows; `mode="startup"` terms fire once. Fusion (brief §4)
 * only changes how many graphs exist — dispatch here still gates *every* call,
 * fused or not, so a quiet frame costs no `ort.run()`.
 */
export class EventManager {
  private resetTerms: ResetEntry[] = [];
  private intervalTerms: Array<{ term: OnnxEvent; trigger: IntervalTrigger }> = [];
  private startupTerms: Array<{ term: OnnxEvent; trigger: StartupTrigger }> = [];
  /** Model-field randomizations, applied once by `startup()` (see `modelFieldDr`). */
  private modelFieldTerms: Array<{ config: ModelFieldDrConfig; trigger: StartupTrigger }> = [];

  constructor(
    configs: EventConfig[],
    registry: Record<string, EventConstructor>,
    private readonly deps?: EventManagerDeps,
  ) {
    for (const config of configs) {
      if (isModelFieldDrConfig(config)) {
        // Perturbs `mjModel` rather than running a graph, so it needs no session —
        // just the seeded PRNG, which `startup()` supplies.
        this.modelFieldTerms.push({ config, trigger: new StartupTrigger() });
        continue;
      }
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
      const EventClass = registry[config.name];
      if (!EventClass) {
        console.warn(`[EventManager] Unknown event type: ${config.name}`);
        continue;
      }
      this.resetTerms.push({ kind: 'plugin', term: new EventClass(config), trigger: new ResetTrigger() });
    }
  }

  /** Fire every `mode="startup"` term once. Call after the scene/policy loads. */
  async startup(context: EventContext): Promise<void> {
    this.applyModelFieldTerms(context);
    await Promise.all(
      this.startupTerms.filter(({ trigger }) => trigger.take()).map(({ term }) => term.fire(context)),
    );
  }

  /**
   * Apply the model-field randomizations, once.
   *
   * Synchronous and before the graph-backed startup terms: `add`/`scale` are
   * relative to the compiled default, which is only still in the model until
   * something writes it.
   */
  private applyModelFieldTerms(context: EventContext): void {
    if (this.modelFieldTerms.length === 0) return;
    const { mujoco, mjModel, mjData } = context;
    if (!mujoco || !mjModel || !mjData) {
      console.warn('[EventManager] model-field randomization needs a live model; skipping.');
      return;
    }
    if (!this.deps?.rng) {
      console.warn('[EventManager] model-field randomization needs the seeded rng; skipping.');
      return;
    }
    // One `defaults` for the whole pass, so several events on one field all read the
    // same compiled base rather than each other's output.
    const defaults = new ModelFieldDefaults(mjModel);
    for (const { config, trigger } of this.modelFieldTerms) {
      if (!trigger.take()) continue;
      applyModelFieldDr(mujoco, mjModel, mjData, config, this.deps.rng, defaults);
    }
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
