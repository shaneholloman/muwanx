import { ObservationBase, type ObservationConfig } from './ObservationBase';
import type { PolicyState } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';
import { NodeStateStore, evaluateGraph } from '../dsl/interpreter';
import type { DslGraph } from '../dsl/types';

/**
 * Wraps a DSL composition graph as an observation term (see ADR 0003).
 *
 * The graph is data passed in `config.graph`; this class is the only TS
 * implementation needed regardless of the term's specifics.  Adding new
 * declarative observations no longer requires an engine PR — only growing
 * the primitive registry does.
 *
 * Output shape is inferred at first call from the graph's output value
 * (scalar → length 1, Float32Array → its length).  History/scale/clip are
 * handled by the surrounding observation-group pipeline, not here.
 */
export class DslObservation extends ObservationBase {
  private readonly graph: DslGraph;
  private readonly params: Record<string, unknown>;
  private readonly store = new NodeStateStore();
  private cachedSize: number | null = null;

  constructor(
    runner: PolicyRunner,
    config: ObservationConfig & { graph: DslGraph; params?: Record<string, unknown> },
  ) {
    super(runner, config);
    this.graph = config.graph;
    this.params = config.params ?? {};
  }

  get size(): number {
    return this.cachedSize ?? this.inferSize();
  }

  reset(_state?: PolicyState): void {
    this.store.reset();
  }

  compute(state: PolicyState): Float32Array {
    const raw = this.evaluate(state);
    // Fix the observation width at the first evaluation (the policy network's
    // expected input size) and conform every later output to it. Some graphs
    // legitimately vary length between frames — e.g. a tracking command whose
    // vector empties (length 0) while its motion is deselected and grows back
    // on re-select. Without a fixed width, the group buffer is sized from the
    // previous frame and `set()` overflows (RangeError) when the vector grows.
    if (this.cachedSize === null) {
      this.cachedSize = raw.length;
      return raw;
    }
    if (raw.length === this.cachedSize) {
      return raw;
    }
    const out = new Float32Array(this.cachedSize);
    out.set(raw.subarray(0, Math.min(raw.length, this.cachedSize)));
    return out;
  }

  /** Evaluate the graph and coerce its result to a Float32Array. */
  private evaluate(state: PolicyState): Float32Array {
    const result = evaluateGraph(
      this.graph,
      { runner: this.runner, state, params: this.params },
      this.store,
    );
    if (result instanceof Float32Array) {
      return result;
    }
    if (typeof result === 'number') {
      return new Float32Array([result]);
    }
    if (typeof result === 'boolean') {
      return new Float32Array([result ? 1 : 0]);
    }
    if (result instanceof Uint8Array) {
      const out = new Float32Array(result.length);
      for (let i = 0; i < result.length; i++) out[i] = result[i];
      return out;
    }
    return new Float32Array(0);
  }

  /**
   * Size inference fires before the first `compute()` call when the
   * observation group needs to allocate buffers.  Run the graph against
   * the current state once; the result is cached.
   */
  private inferSize(): number {
    const context = this.runner.getContext();
    const fakeState: PolicyState = {
      jointPos: new Float32Array(this.runner.getNumActions()),
    };
    void context;
    // Evaluate and discard the result; `compute` updated cachedSize as a side
    // effect.
    this.compute(fakeState);
    return this.cachedSize ?? 0;
  }
}
