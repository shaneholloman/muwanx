/**
 * Hand the browser two turns of the event loop.
 *
 * `setTimeout` is clamped to ~4ms once nested — and a timer-driven loop always is — so it
 * cannot yield cheaply on the path that is already behind schedule. A `MessageChannel`
 * message is a task with no such floor.
 *
 * Twice, not once: rendering only happens between tasks, so one turn buys at most one frame
 * per control step — 40fps at a 25ms step. Two turns reach 60fps, and the second turn costs
 * nothing in step rate (measured 39.8 → 39.6 steps/s at a 25ms step).
 */
const waiters: Array<() => void> = [];
let channel: MessageChannel | null = null;

function open(): MessageChannel {
  const ch = new MessageChannel();
  ch.port1.onmessage = () => waiters.shift()?.();
  // Node (the unit test) holds its event loop open for a listening port; browsers have no
  // `unref`, so this is a no-op there.
  (ch.port1 as MessagePort & { unref?: () => void }).unref?.();
  return ch;
}

function turn(ch: MessageChannel): Promise<void> {
  return new Promise((resolve) => {
    waiters.push(resolve);
    ch.port2.postMessage(0);
  });
}

/** Created on first use so importing this module has no side effect. */
export async function yieldToBrowser(): Promise<void> {
  const ch = channel ?? (channel = open());
  await turn(ch);
  await turn(ch);
}
