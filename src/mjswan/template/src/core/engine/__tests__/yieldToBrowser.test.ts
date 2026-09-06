import { describe, expect, it } from 'vitest';

import { yieldToBrowser } from '../yieldToBrowser';

describe('yieldToBrowser', () => {
  // Guards issue #103: `await Promise.resolve()` looks like a yield but never reaches the event loop.
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
