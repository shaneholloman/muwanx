/**
 * ORT-Web's active-run slot is module-global: a second `run()` started before the
 * first resolves throws `Session already started`, whatever session it belongs to.
 * `queueOrtRun` is what keeps the policy net, the fused graphs and the reset-time
 * event graphs from overlapping — so what it must guarantee is exactly that no two
 * queued runs are ever in flight at once.
 */
import { describe, expect, it } from 'vitest';

import { queueOrtRun } from '../runQueue';

/** A run that reports whether it ever overlapped another. */
function overlapTracker() {
  let inFlight = 0;
  let overlapped = false;
  return {
    get overlapped() {
      return overlapped;
    },
    run: async <T>(value: T): Promise<T> => {
      inFlight += 1;
      if (inFlight > 1) overlapped = true;
      await Promise.resolve();
      await Promise.resolve();
      inFlight -= 1;
      return value;
    },
  };
}

describe('queueOrtRun', () => {
  it('never lets two runs overlap, and keeps their results in order', async () => {
    const tracker = overlapTracker();
    const results = await Promise.all(
      [1, 2, 3, 4].map(n => queueOrtRun(() => tracker.run(n))),
    );
    expect(tracker.overlapped).toBe(false);
    expect(results).toEqual([1, 2, 3, 4]);
  });

  it('runs the next one after a failure', async () => {
    // A failed run still released ORT's slot; inheriting the rejection would wedge
    // the queue for the rest of the session.
    const failed = queueOrtRun(() => Promise.reject(new Error('boom')));
    await expect(failed).rejects.toThrow('boom');
    await expect(queueOrtRun(() => Promise.resolve('next'))).resolves.toBe('next');
  });
});
