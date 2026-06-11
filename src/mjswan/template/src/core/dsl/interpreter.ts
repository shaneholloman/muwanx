/**
 * DSL composition-graph interpreter (see ADR 0003).
 *
 * Evaluates a graph once per step: each node's primitive op runs against the
 * resolved values of its `in` edges and produces a value bound to its `out`
 * edge.  The graph is data; the interpreter is the only code on the per-sim
 * path.
 *
 * Stateful primitives (e.g. `StepCount`, `History`) hold per-node state
 * across steps via the `NodeStateStore` an interpreter instance owns;
 * `reset()` clears the store and matches episode-reset semantics.
 */

import type { EvalContext, NodeState } from './primitives';
import { Primitives } from './primitives';
import type { DslGraph, DslPrimitiveValue } from './types';

export type { EvalContext } from './primitives';
export { Primitives } from './primitives';

export class NodeStateStore {
  private states = new Map<string, NodeState>();

  get(outName: string): NodeState {
    let s = this.states.get(outName);
    if (s === undefined) {
      s = {};
      this.states.set(outName, s);
    }
    return s;
  }

  reset(): void {
    this.states.clear();
  }
}

export function evaluateGraph(
  graph: DslGraph,
  ctx: EvalContext,
  store: NodeStateStore,
): DslPrimitiveValue {
  const values: Record<string, DslPrimitiveValue> = {};
  for (const node of graph.nodes) {
    const prim = Primitives[node.op];
    if (prim === undefined) {
      throw new Error(
        `DSL interpreter: unknown primitive op "${node.op}". `
        + `Add it to the engine's primitive registry, or fix the build's serializer.`,
      );
    }
    const inputs: DslPrimitiveValue[] = (node.in ?? []).map((name) => {
      if (!(name in values)) {
        throw new Error(
          `DSL interpreter: node "${node.out}" references unresolved input "${name}". `
          + 'Graph nodes must appear in topological order.',
        );
      }
      return values[name];
    });
    values[node.out] = prim(inputs, node.attrs ?? {}, ctx, store.get(node.out));
  }
  const result = values[graph.output];
  if (result === undefined) {
    throw new Error(
      `DSL interpreter: graph output "${graph.output}" was not produced by any node.`,
    );
  }
  return result;
}

export function isDslGraph(entry: unknown): entry is DslGraph {
  return (
    typeof entry === 'object'
    && entry !== null
    && 'nodes' in entry
    && 'output' in entry
    && Array.isArray((entry as { nodes: unknown }).nodes)
  );
}
