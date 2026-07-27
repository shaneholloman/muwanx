/**
 * Finding the traced term graphs a build refers to (ADR 0005 §4).
 *
 * Under ADR 0005 a policy is no longer one `.onnx`: each observation,
 * termination, command and event term whose body was traced gets its own small
 * graph, written by the Builder next to `policy.json` as `obs/<name>.onnx`,
 * `term/…`, `command/…`, `event/…` and referenced from the config by that path.
 *
 * The engine takes bytes and never fetches (ADR 0004 §4), so someone has to turn
 * those paths into bytes before `loadScene`. That someone needs the list of
 * paths, and the list is a property of the config — so it is derived here, once,
 * rather than re-implemented by every consumer (the in-repo app via
 * `mjswan/manifest`, mjswan Cloud, and the test harness).
 *
 * Deliberately structural, not schema-aware: collect every `onnx` string under
 * the sections that hold term entries. A new term kind then needs no change
 * here, and a missing graph is caught where it matters — the manager that wanted
 * a session for it warns by name and skips the term.
 */

/** Sections of a policy config whose entries may carry a traced graph. */
const POLICY_SECTIONS = ['observations', 'terminations', 'commands'] as const;

function collectFrom(value: unknown, into: Set<string>): void {
  if (Array.isArray(value)) {
    for (const item of value) collectFrom(item, into);
    return;
  }
  if (typeof value !== 'object' || value === null) return;
  const record = value as Record<string, unknown>;
  // `onnx` names a per-term graph; `fused` names a whole group's graph (ADR 0005
  // §4). Both are paths relative to the config. `onnx` is also an *object* at the
  // policy top level (`{path, meta}` — the policy network itself, which arrives as
  // `PolicyInput.onnx`), hence the string check.
  for (const key of ['onnx', 'fused'] as const) {
    const ref = record[key];
    if (typeof ref === 'string') into.add(ref);
  }
  for (const nested of Object.values(record)) collectFrom(nested, into);
}

/**
 * Traced-graph paths referenced by a `policy.json`, relative to its own
 * directory. Sorted, so a caller's load order and logging are deterministic.
 */
export function policyGraphRefs(config: Record<string, unknown>): string[] {
  const refs = new Set<string>();
  for (const section of POLICY_SECTIONS) collectFrom(config[section], refs);
  return [...refs].sort();
}

/** Traced-graph paths referenced by a scene's event configs (`config.json`). */
export function eventGraphRefs(events: unknown): string[] {
  const refs = new Set<string>();
  collectFrom(events, refs);
  return [...refs].sort();
}
