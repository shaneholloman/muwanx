/**
 * One queue for every ORT-Web inference in the page.
 *
 * ORT-Web's wasm binding keeps a *module-global* active-run slot: overlapping
 * `run()` calls throw `Session already started` / `Session mismatch`, even across
 * different sessions. The engine has several independent async ORT callers, so all
 * of them go through here. Costs nothing — the wasm backend is single-threaded.
 */
let tail: Promise<unknown> = Promise.resolve();

/** Run `inference` once the previous queued run has settled. */
export function queueOrtRun<T>(inference: () => Promise<T>): Promise<T> {
  // Same callback on both paths: a failed run still released ORT's slot.
  const result = tail.then(inference, inference);
  tail = result.catch(() => undefined);
  return result;
}
