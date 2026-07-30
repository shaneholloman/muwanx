/**
 * One queue for every ORT-Web inference in the page.
 *
 * ORT-Web's wasm binding keeps a *module-global* active-run slot, not a per-session
 * one: starting a second `run()` before the first resolves throws `Session already
 * started`, and the first then finishes into `Session mismatch`. Different sessions
 * collide just the same. The engine has several independent async ORT callers — the
 * policy network on the step loop, the fused observation/termination graphs, and
 * the event/command graphs that fire on reset (`TrackingCommand`'s RSI jitter is
 * deliberately fire-and-forget, so it *will* overlap a step) — so every one of them
 * has to go through here.
 *
 * Serializing costs nothing real: the wasm backend is single-threaded
 * (`numThreads = 1`), so these runs never executed in parallel anyway.
 */
let tail: Promise<unknown> = Promise.resolve();

/** Run `inference` once the previous queued run has settled. */
export function queueOrtRun<T>(inference: () => Promise<T>): Promise<T> {
  // Same callback on both paths: a failed run still released ORT's slot, so the
  // next one must go ahead rather than inherit the rejection.
  const result = tail.then(inference, inference);
  tail = result.catch(() => undefined);
  return result;
}
