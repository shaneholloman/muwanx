/**
 * `OnnxTermination`: a termination term whose body is a traced ONNX graph
 * (ADR 0005 §1). One generic class covers every traced termination — the graph
 * and its declared input slots are data in `policy.json`.
 *
 * The graph's output is a bool (mjlab's terms return a bool tensor), carried over
 * the wire as ORT's `bool` dtype, i.e. a `Uint8Array` of 0/1. Any non-zero
 * element means "done", matching mjlab's per-env semantics reduced to the single
 * environment this runtime targets (ADR §5).
 *
 * **Async boundary.** `TerminationBase.evaluate()` is synchronous, but ORT is not.
 * Rather than block the step loop, this kicks off inference and reports the most
 * recently completed verdict — a frame that arrives while inference is in flight
 * is skipped, not queued, exactly as `OnnxCommand`/`OnnxEvent` do. A one-frame
 * late reset is the accepted lag (ADR §8); an unbounded queue would be worse.
 */

import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import { slotInputName } from '../onnx/session';
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
        // Cannot evaluate on absent state; hold the previous verdict rather than
        // reporting "not done" and letting a real termination slip through.
        console.warn(
          `[OnnxTermination] "${this.config.name}" could not read slot ` +
            `${slotInputName(slot)}; holding the previous verdict.`,
        );
        return;
      }
      feeds[slotInputName(slot)] = { data: value, dims: [1, value.length] };
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
