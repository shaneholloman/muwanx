import { describe, expect, it } from 'vitest';

import { yieldToBrowser } from '../yieldToBrowser';

describe('yieldToBrowser', () => {
  // The bug it guards (issue #103): a microtask-only "yield" — `await Promise.resolve()`,
  // `queueMicrotask` — never returns to the event loop, so rendering never runs.
  it('resolves on a later task, after the microtask queue drains', async () => {
    const order: string[] = [];
    const yielded = yieldToBrowser().then(() => order.push('task'));
    await Promise.resolve().then(() => order.push('microtask'));
    await yielded;
    expect(order).toEqual(['microtask', 'task']);
  });

  it('resolves every caller once, in call order', async () => {
    const seen: number[] = [];
    await Promise.all([0, 1, 2].map((i) => yieldToBrowser().then(() => seen.push(i))));
    expect(seen).toEqual([0, 1, 2]);
  });
});
