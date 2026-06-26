import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { PolicyState } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';
import { NodeStateStore, evaluateGraph } from '../dsl/interpreter';
import type { DslGraph } from '../dsl/types';

/**
 * Wraps a DSL composition graph as a termination term (see ADR 0003).
 *
 * The graph is data passed in `config.graph`; this class is the only TS
 * implementation needed regardless of the term's specifics.  Adding new
 * declarative terminations no longer requires an engine PR — only growing
 * the primitive registry does.
 *
 * Stateful primitives (e.g. `StepCount`) keep per-node state in the
 * `NodeStateStore`, which `reset()` clears to match episode boundaries.
 */
export class DslTermination extends TerminationBase {
  private readonly graph: DslGraph;
  private readonly runner: PolicyRunner;
  private readonly params: Record<string, unknown>;
  private readonly store = new NodeStateStore();

  constructor(runner: PolicyRunner, config: TerminationConfig & { graph: DslGraph }) {
    super(config);
    this.runner = runner;
    this.graph = config.graph;
    this.params = config.params ?? {};
  }

  evaluate(state: PolicyState): boolean {
    const result = evaluateGraph(
      this.graph,
      { runner: this.runner, state, params: this.params },
      this.store,
    );
    if (typeof result === 'boolean') return result;
    if (typeof result === 'number') return result !== 0;
    return false;
  }

  reset(): void {
    this.store.reset();
  }
}
