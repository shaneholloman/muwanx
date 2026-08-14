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

/** A plugin-supplied event class — reset-only, no traced graph. */
type PluginTerm = { kind: 'plugin'; term: EventBase; trigger: ResetTrigger };
type OnnxResetTerm = { kind: 'onnx'; term: OnnxEvent; trigger: ResetTrigger };
type ResetEntry = PluginTerm | OnnxResetTerm;

/**
 * Mode-aware event dispatch: `mode="reset"` terms fire on episode reset behind a
 * `ResetTrigger`, `mode="interval"` on the frames their `IntervalTrigger` allows,
 * `mode="startup"` once.
 *
 * One graph per term, unfused — the reference tasks have at most two reset terms and
 * none of the other modes, and fusing would have to reproduce the config-order write
 * semantics `onReset` gets for free.
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
        // Perturbs `mjModel` rather than running a graph, so it needs no session.
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
      if (config.native) {
        // The build declined to trace this term and said why; not a missing plugin.
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

  /**
   * Fire every `mode="startup"` term once, in config order and after the model-field
   * randomizations, so `add`/`scale` see the compiled default.
   */
  async startup(context: EventContext): Promise<void> {
    this.applyModelFieldTerms(context);
    for (const { term, trigger } of this.startupTerms) {
      if (trigger.take()) await term.fire(context);
    }
  }

  /** Once, before the startup terms: `add`/`scale` are relative to the compiled default. */
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
    // One `defaults` per pass, so events on one field share a base.
    const defaults = new ModelFieldDefaults(mjModel);
    for (const { config, trigger } of this.modelFieldTerms) {
      if (!trigger.take()) continue;
      applyModelFieldDr(mujoco, mjModel, mjData, config, this.deps.rng, defaults);
    }
  }

  /**
   * Advance one control step: fire the `mode="interval"` terms whose timer elapsed **in
   * config order**, and advance every reset-gate counter. Sequential, so overlaps
   * resolve by config order rather than by ORT completion; the counters go first, since
   * a failing graph must not stall them.
   */
  async tick(dt: number, context: EventContext): Promise<void> {
    for (const { trigger } of this.resetTerms) trigger.step();
    for (const { term, trigger } of this.intervalTerms) {
      if (trigger.tick(dt)) await term.fire(context);
    }
  }

  /**
   * Fire every `mode="reset"` term whose gate allows it, **in config order**. Sequential
   * rather than `Promise.all`: every write is an assignment over `default_*`, so the
   * order decides which value survives an overlap.
   */
  async onReset(context: EventContext): Promise<void> {
    for (const entry of this.resetTerms) {
      if (!entry.trigger.take()) continue;
      if (entry.kind === 'onnx') await entry.term.fire(context);
      else entry.term.onReset(context);
    }
    for (const { trigger } of this.intervalTerms) trigger.onReset();
  }

  get size(): number {
    return (
      this.resetTerms.length +
      this.intervalTerms.length +
      this.startupTerms.length +
      // Counted: a task whose only events are DR would otherwise report "0 loaded".
      this.modelFieldTerms.length
    );
  }
}
