/**
 * `OnnxEvent`: the generic ONNX-backed event handler (ADR 0005 §3/§4, brief §3).
 *
 * Every traced event term — `push_robot`, `reset_slider`, `reset_base`, … — is a
 * data instantiation of this one class, symmetric to `OnnxCommand`. Unlike a
 * Command, the events traced so far carry **no persistent state across frames**
 * (a reset/interval event body is a pure function of constants baked into the
 * graph, whatever `input_slots` it reads live, and `rand`); should a future
 * event need state, that is an extension of this class, not a new one.
 *
 * What this class owns (the native half; the graph owns the math):
 *
 * - **`rand`** — drawn from the orchestrator-owned seeded PRNG (ADR §2).
 * - **`entity_write`** — hands the graph's computed pose/velocity/joint-state to
 *   the apply primitive (`applyEntityWrites`).
 *
 * *Timing* (when this fires) is owned by the caller — `EventManager` drives one
 * of `IntervalTrigger` / `StartupTrigger` / `ResetTrigger` and calls `fire()`
 * only on the frames/resets a trigger allows. Fusion (brief §4) reduces how many
 * graphs exist, never how often a firing term is called.
 *
 * **Async boundary.** `fire()` is async (ORT-Web inference). Interval/startup
 * dispatch is fire-and-forget with an in-flight guard (mirrors `OnnxCommand`) so
 * a slow graph can never queue a backlog of the same disturbance. Reset-mode
 * firings are different: the caller (`EventManager.onReset`) awaits them,
 * because the reset must be visibly complete before the next frame renders
 * (ADR 0005 §8's synchronous reset-then-forward step-loop ordering).
 */

import { SeededRng } from '../rng';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import { applyEntityWrites, type WriteTarget, type WriteValues } from './entityWrite';
import type { EventContext } from './EventBase';

export type EventMode = 'startup' | 'reset' | 'interval';

export interface OnnxEventConfig {
  name: string;
  mode: EventMode;
  onnx: string;
  rand_dim: number;
  input_slots?: OnnxInputSlot[];
  write_targets?: WriteTarget[];
  rand_ranges?: Array<[number, number]>;
  /** `mode="interval"` only: `[min, max]` seconds between firings. */
  interval_range_s?: [number, number];
  /** `mode="interval"` only: timer survives episode reset when true. */
  is_global_time?: boolean;
  /** `mode="reset"` only: suppress firing on resets that arrive too soon. */
  min_step_count_between_reset?: number;
}

export function isOnnxEventConfig(config: unknown): config is OnnxEventConfig {
  return (
    typeof config === 'object'
    && config !== null
    && typeof (config as { onnx?: unknown }).onnx === 'string'
    && typeof (config as { mode?: unknown }).mode === 'string'
  );
}

export interface OnnxEventDeps {
  session: OnnxSession;
  rng: SeededRng;
  readSlot?: SlotReader;
}

export class OnnxEvent {
  readonly name: string;
  readonly mode: EventMode;
  private inFlight = false;

  constructor(
    readonly config: OnnxEventConfig,
    private readonly deps: OnnxEventDeps,
  ) {
    this.name = config.name;
    this.mode = config.mode;
  }

  /** True while a `fire()` is in flight (callers use this to skip, not queue). */
  get busy(): boolean {
    return this.inFlight;
  }

  /** Run the graph once and apply any `entity_write` output. Never throws into the caller's tick loop for a missing model/data — it is the context's job to supply them. */
  async fire(context: EventContext): Promise<void> {
    if (this.inFlight) return;
    this.inFlight = true;
    try {
      const feeds: Record<string, OnnxTensorLike> = {};
      for (const slot of this.config.input_slots ?? []) {
        const value = this.deps.readSlot?.(slot) ?? null;
        if (!value) continue;
        feeds[`${slot.entity ?? 'entity'}__${slot.field}`] = { data: value, dims: [1, value.length] };
      }
      feeds.rand = {
        data: this.deps.rng.randVector(this.config.rand_dim, this.config.rand_ranges),
        dims: [this.config.rand_dim],
      };

      const outputs = await this.deps.session.run(feeds);
      this.applyWrites(context, outputs);
    } finally {
      this.inFlight = false;
    }
  }

  private applyWrites(context: EventContext, outputs: Record<string, OnnxTensorLike>): void {
    const targets = this.config.write_targets ?? [];
    if (targets.length === 0) return;
    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return;
    const values: WriteValues = {};
    for (const [key, tensor] of Object.entries(outputs)) values[key] = toFloat32(tensor.data);
    applyEntityWrites(mjModel, mjData, targets, values);
  }
}

function toFloat32(data: Float32Array | BigInt64Array | Uint8Array): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Number(data[i]);
  return out;
}
