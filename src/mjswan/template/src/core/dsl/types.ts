/**
 * DSL composition-graph types (see ADR 0003).
 *
 * The build emits this JSON shape into `policy.json` for declarative MDP
 * terms; the engine's interpreter walks it once per step (per term).  Every
 * `op` must be a key in the engine's primitive registry; every `in` and
 * `out` is a string-named edge inside the graph.
 */

export type DslPrimitiveValue = number | Float32Array | boolean | Uint8Array;

export type DslNode = {
  op: string;
  out: string;
  in?: string[];
  attrs?: Record<string, unknown>;
};

export type DslGraph = {
  kind: 'observation' | 'termination';
  nodes: DslNode[];
  output: string;
};
