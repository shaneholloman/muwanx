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
 * `IntervalTrigger` allows; `mode="startup"` terms fire once. Events are one graph
 * per term and stay that way: per-mode fusion was measured and declined (brief §4b)
 * — no reference task has a traced `startup` or `interval` term at all, `reset` has
 * at most two, and it would have to reproduce the write semantics `onReset` gets for
 * free by looping in config order (last writer wins per element, as mjlab does).
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

  /**
   * Fire every `mode="startup"` term once. Call after the scene/policy loads.
   *
   * In config order, for the same reason `onReset` is (mjlab loops its terms), and
   * after the model-field randomizations so `add`/`scale` see the compiled default.
   */
  async startup(context: EventContext): Promise<void> {
    this.applyModelFieldTerms(context);
    for (const { term, trigger } of this.startupTerms) {
      if (trigger.take()) await term.fire(context);
    }
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
    // Fire-and-forget, unlike `onReset`/`startup`: `tick` is synchronous because the
    // step loop is, so there is nothing to await into. Two interval terms writing the
    // same element would therefore resolve in completion order rather than mjlab's
    // config order — no reference task has even one traced interval term, so this is
    // noted rather than solved.
    for (const { term, trigger } of this.intervalTerms) {
      if (trigger.tick(dt)) void term.fire(context);
    }
    for (const { trigger } of this.resetTerms) trigger.step();
  }

  /**
   * Fire every `mode="reset"` term whose gate allows it, **in config order**.
   *
   * Sequential rather than `Promise.all`, to match mjlab: its
   * `EventManager.apply` loops the mode's terms in order, and every write is an
   * assignment (`data.qpos[env_ids, q_slice] = position`), so two terms touching
   * the same element resolve last-writer-wins by config order. Firing them
   * concurrently made that resolution order instead — harmless while writes are
   * disjoint, which they are on every reference task, and nondeterministic the
   * moment they overlap.
   *
   * The term bodies read `default_*` rather than live state (mjlab's
   * `reset_joints_by_offset` starts from `default_joint_pos`), so sequencing does
   * not make them compound; it only decides which value survives on an overlap.
   *
   * Costs one reset's worth of serialized inference, which is a rare frame.
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
    return this.resetTerms.length + this.intervalTerms.length + this.startupTerms.length;
  }
}
