/**
 * The generic ONNX-backed event handler, symmetric to `OnnxCommand`: every traced event
 * term is a data instantiation of this class. Unlike a command, an event body is a pure
 * function of its constants, `input_slots` and `rand` — no state across frames.
 *
 * The graph owns the math; this draws `rand` from the seeded PRNG and applies the
 * `entity_write` output. *When* it fires is the caller's business (`EventManager`).
 *
 * **Async boundary.** Interval/startup dispatch is fire-and-forget with an in-flight
 * guard, so a slow graph cannot queue a backlog of the same disturbance. Reset firings
 * are awaited, since the reset must be complete before the next frame renders.
 */

import { SeededRng } from '../rng';
import { buildFeeds, toFloat32 } from '../onnx/session';
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

  /** Run the graph once and apply any `entity_write` output. */
  async fire(context: EventContext): Promise<void> {
    if (this.inFlight) return;
    this.inFlight = true;
    try {
      const { feeds } = buildFeeds(this.config.input_slots, this.deps.readSlot);
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
