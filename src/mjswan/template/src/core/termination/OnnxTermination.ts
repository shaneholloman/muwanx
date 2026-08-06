/**
 * A termination term whose body is a traced ONNX graph; one generic class covers every
 * one, since the graph and its input slots are data in `policy.json`.
 *
 * The output is ORT's `bool` dtype — a `Uint8Array` of 0/1 — and any non-zero element
 * means done, mjlab's per-env semantics at N=1.
 *
 * **Async boundary.** `evaluate()` is sync while ORT is not, so this kicks off inference
 * and reports the last completed verdict; a frame arriving mid-flight is skipped rather
 * than queued.
 */

import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import { slotDims, slotInputName } from '../onnx/session';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

export interface OnnxTerminationConfig extends TerminationConfig {
  onnx: string;
  input_slots?: OnnxInputSlot[];
}

export interface OnnxTerminationDeps {
  session: OnnxSession;
  readSlot: SlotReader;
}

export class OnnxTermination extends TerminationBase {
  private readonly onnxConfig: OnnxTerminationConfig;
  private readonly deps: OnnxTerminationDeps;
  private done = false;
  private inFlight = false;

  constructor(
    runner: PolicyRunner,
    config: OnnxTerminationConfig,
    deps: OnnxTerminationDeps,
  ) {
    super(runner, config);
    this.onnxConfig = config;
    this.deps = deps;
  }

  evaluate(_state: PolicyState): boolean {
    if (!this.inFlight) {
      this.inFlight = true;
      void this.step().finally(() => {
        this.inFlight = false;
      });
    }
    return this.done;
  }

  reset(): void {
    this.done = false;
  }

  /** Run one graph evaluation. Exposed for tests / deterministic stepping. */
  async step(): Promise<void> {
    const feeds: Record<string, OnnxTensorLike> = {};
    for (const slot of this.onnxConfig.input_slots ?? []) {
      const value = this.deps.readSlot(slot);
      if (!value) {
        // Hold the previous verdict rather than letting a termination slip through.
        console.warn(
          `[OnnxTermination] "${this.config.name}" could not read slot ` +
            `${slotInputName(slot)}; holding the previous verdict.`,
        );
        return;
      }
      feeds[slotInputName(slot)] = { data: value, dims: slotDims(slot, value.length) };
    }
    const outputs = await this.deps.session.run(feeds);
    const first = Object.values(outputs)[0];
    if (!first) {
      console.warn(`[OnnxTermination] "${this.config.name}" produced no output.`);
      return;
    }
    this.done = isAnyTruthy(first.data);
  }
}

function isAnyTruthy(data: Float32Array | BigInt64Array | Uint8Array): boolean {
  for (let i = 0; i < data.length; i++) {
    if (Number(data[i]) !== 0) return true;
  }
  return false;
}
