/**
 * Per-term observation history (mjlab's `ObservationTermCfg.history_length`).
 *
 * mjlab stacks each term's frames *before* concatenating the group, so history
 * cannot live at the group level for a group where only some terms carry it — the
 * group-level ring buffer in `PolicyRunner` gives step-major order over the whole
 * concatenated vector, where mjlab gives term-major. This wraps a single term
 * instead, which is also why a group with per-term history does not fuse
 * (`_group_is_fusable`): the stacking happens outside the graph.
 *
 * `offsets` generalises the count: a dense `history_length: n` arrives as
 * `[0..n-1]`, and a policy trained on *sparse* look-back (the offsets that show up
 * in tracking policies, e.g. `[0, 1, 2, 4, 8, 16]`) names them directly. The buffer
 * always holds `max(offsets) + 1` frames; only the named ones reach the output.
 */

import { ObservationBase, type ObservationConfig } from './ObservationBase';
import type { PolicyState } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';

/** Offsets a term's history is sampled at, or null when it keeps no history. */
export function historyOffsets(entry: ObservationConfig): number[] | null {
  const sparse = (entry as { history_offsets?: unknown }).history_offsets;
  if (Array.isArray(sparse)) {
    const offsets = sparse.map((value) => Math.max(0, Math.trunc(Number(value) || 0)));
    return offsets.length > 0 ? offsets : null;
  }
  const length = Math.trunc(Number((entry as { history_length?: unknown }).history_length) || 0);
  if (length <= 1) return null;
  return Array.from({ length }, (_, i) => i);
}

export class HistoryObservation extends ObservationBase {
  private readonly base: ObservationBase;
  private readonly offsets: number[];
  private readonly interleaved: boolean;
  private readonly frames: Float32Array[];
  /** Set by `reset()`: the next frame fills every slot instead of shifting in. */
  private needsPrime = true;

  constructor(
    runner: PolicyRunner,
    config: ObservationConfig,
    base: ObservationBase,
    offsets: number[],
  ) {
    super(runner, config);
    this.base = base;
    this.offsets = offsets;
    this.interleaved = Boolean((config as { history_interleaved?: unknown }).history_interleaved);
    this.frames = Array.from(
      { length: Math.max(...offsets) + 1 },
      () => new Float32Array(base.size),
    );
  }

  get size(): number {
    return this.base.size * this.offsets.length;
  }

  reset(state?: PolicyState): void {
    this.base.reset?.(state);
    this.needsPrime = true;
  }

  update(state: PolicyState): void {
    this.base.update?.(state);
  }

  preload(): Promise<void> {
    return this.base.preload?.() ?? Promise.resolve();
  }

  async compute(state: PolicyState): Promise<Float32Array> {
    const frame = Float32Array.from(await this.base.compute(state));
    if (this.needsPrime) {
      // First frame after a reset: every slot is this frame, so the policy never
      // sees a history of zeros it was not trained on (same rule as the
      // group-level buffer in `PolicyRunner`).
      for (const slot of this.frames) slot.set(frame.subarray(0, slot.length));
      this.needsPrime = false;
    } else {
      // Shift by rotating the array rather than copying `max(offsets)` frames:
      // the oldest buffer becomes this frame's slot.
      const oldest = this.frames.pop();
      if (oldest) {
        oldest.set(frame.subarray(0, oldest.length));
        this.frames.unshift(oldest);
      }
    }
    return this.gather();
  }

  /** Frame-major (`[frame_t, frame_{t-1}, …]`), or element-major when interleaved. */
  private gather(): Float32Array {
    const width = this.base.size;
    const out = new Float32Array(width * this.offsets.length);
    for (let i = 0; i < this.offsets.length; i++) {
      const frame = this.frames[Math.min(this.offsets[i], this.frames.length - 1)];
      if (!frame) continue;
      if (this.interleaved) {
        for (let j = 0; j < width; j++) out[j * this.offsets.length + i] = frame[j];
      } else {
        out.set(frame.subarray(0, width), i * width);
      }
    }
    return out;
  }
}
