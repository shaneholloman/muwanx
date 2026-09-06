/**
 * Hand the browser two turns of the event loop.
 *
 * `MessageChannel`, not `setTimeout`: nested timers are clamped to ~4ms, a floor on exactly
 * the path already behind schedule. Two turns, not one: rendering happens between tasks, so
 * one turn buys at most one frame per control step.
 */
const waiters: Array<() => void> = [];
let channel: MessageChannel | null = null;

function open(): MessageChannel {
  const ch = new MessageChannel();
  ch.port1.onmessage = () => waiters.shift()?.();
  // Node keeps its event loop alive for a listening port; browsers have no `unref` to call.
  (ch.port1 as MessagePort & { unref?: () => void }).unref?.();
  return ch;
}

function turn(ch: MessageChannel): Promise<void> {
  return new Promise((resolve) => {
    waiters.push(resolve);
    ch.port2.postMessage(0);
  });
}

/** The channel is opened on first use, so importing this module has no side effect. */
export async function yieldToBrowser(): Promise<void> {
  const ch = channel ?? (channel = open());
  await turn(ch);
  await turn(ch);
}
