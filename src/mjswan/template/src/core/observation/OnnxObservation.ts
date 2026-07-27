/**
 * `OnnxObservation`: an observation term whose body is a traced ONNX graph
 * (ADR 0005 §1). One generic class covers every traced observation — the graph,
 * its declared input slots, and the native clip/scale metadata are all data in
 * `policy.json`, so adding an observation never touches the engine.
 *
 * **Why `size` comes from the build.** `PolicyRunner` needs each term's width
 * synchronously, at load, to lay out the group buffer — but ORT inference is
 * async, so this class cannot measure its own output first. The build knows the
 * width exactly (it is the parity-verified reference output) and ships it, so no
 * speculative inference is needed here.
 *
 * **Async boundary.** Unlike `OnnxCommand`/`OnnxEvent`, which skip a frame rather
 * than queue, observations are on the policy's critical path: a stale observation
 * directly degrades control. So `compute()` returns a promise the group awaits,
 * mirroring ADR 0005 §8's `const obs = await ortObs.run(...)`.
 *
 * Per-term sessions for now; ADR §4 wants the terms of a group fused into one
 * graph, which is a build-time change this class's contract is unaffected by.
 */

import { ObservationBase, type ObservationConfig } from './ObservationBase';
import {
  applyObservationPipeline,
  conformToSize,
  type ObservationClip,
  type ObservationScale,
} from './pipeline';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import { slotInputName } from '../onnx/session';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

export interface OnnxObservationConfig extends ObservationConfig {
  onnx: string;
  /** Output width, from the build (see class docs). */
  size: number;
  input_slots?: OnnxInputSlot[];
  scale?: ObservationScale;
  clip?: ObservationClip;
}

export interface OnnxObservationDeps {
  session: OnnxSession;
  readSlot: SlotReader;
}

/** Whether a config entry names a traced-ONNX observation. */
export function isOnnxObservationConfig(
  entry: ObservationConfig,
): entry is OnnxObservationConfig {
  return typeof (entry as { onnx?: unknown }).onnx === 'string';
}

export class OnnxObservation extends ObservationBase<OnnxObservationConfig> {
  private readonly deps: OnnxObservationDeps;
  /** Last completed value, served if a later frame's inference fails. */
  private last: Float32Array;

  constructor(
    runner: PolicyRunner,
    config: OnnxObservationConfig,
    deps: OnnxObservationDeps,
  ) {
    super(runner, config);
    this.deps = deps;
    this.last = new Float32Array(config.size);
  }

  get size(): number {
    return this.config.size;
  }

  async compute(_state: PolicyState): Promise<Float32Array> {
    const feeds: Record<string, OnnxTensorLike> = {};
    for (const slot of this.config.input_slots ?? []) {
      const value = this.deps.readSlot(slot);
      if (!value) {
        // A slot the runtime cannot supply means the graph would run on stale or
        // absent state; serve the last good value rather than silently feeding
        // zeros into the policy.
        console.warn(
          `[OnnxObservation] "${this.config.name}" could not read slot ` +
            `${slotInputName(slot)}; reusing the previous value.`,
        );
        return this.last;
      }
      feeds[slotInputName(slot)] = { data: value, dims: [1, value.length] };
    }

    const outputs = await this.deps.session.run(feeds);
    const first = Object.values(outputs)[0];
    if (!first) {
      console.warn(`[OnnxObservation] "${this.config.name}" produced no output.`);
      return this.last;
    }
    const raw = toFloat32(first.data);
    this.last = applyObservationPipeline(conformToSize(raw, this.config.size), this.config);
    return this.last;
  }
}

function toFloat32(data: Float32Array | BigInt64Array | Uint8Array): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Number(data[i]);
  return out;
}
