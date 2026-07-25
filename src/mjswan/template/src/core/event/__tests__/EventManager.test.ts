/**
 * `EventManager`: mode-aware Event dispatch (ADR 0005 §5, companion brief §4).
 *
 * Previously `onReset()`-only; these tests pin the new behaviour: interval/
 * startup dispatch is genuinely new (not a port), reset stays backward
 * compatible for legacy DslEvent/registry terms, and a quiet frame never calls
 * `ort.run()` for a term whose trigger did not fire.
 */
import { describe, expect, it, vi } from 'vitest';

import { SeededRng } from '../../rng';
import { OnnxSessionCache, type OnnxSession, type OnnxTensorLike } from '../../onnx/session';
import { EventManager, type EventManagerDeps } from '../EventManager';
import { EventBase, type EventContext } from '../EventBase';

const NO_MODEL: EventContext = { mjModel: null, mjData: null };

/**
 * Drain the microtask queue between synchronous `tick()` calls.
 *
 * `tick()` fires an interval term fire-and-forget; a fake session resolving via
 * `Promise.resolve()` still needs a few microtask turns before the in-flight
 * guard clears. A real mainLoop awaits real time between frames, so this only
 * matters for these tight synchronous test loops.
 */
async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

function fakeSession(run: (feeds: Record<string, OnnxTensorLike>) => Record<string, OnnxTensorLike>): OnnxSession {
  return { run: (feeds) => Promise.resolve(run(feeds)) };
}

async function depsWithSession(
  path: string,
  session: OnnxSession,
  seed = 1,
): Promise<EventManagerDeps> {
  const sessions = new OnnxSessionCache(() => Promise.resolve(session));
  await sessions.load([{ name: path, data: new ArrayBuffer(0) }]);
  return { sessions, rng: new SeededRng(seed) };
}

describe('EventManager: legacy reset-only terms (backward compatible)', () => {
  it('still fires a plain registry-class term on every reset', async () => {
    let calls = 0;
    class Spy extends EventBase {
      onReset(): void {
        calls++;
      }
    }
    const mgr = new EventManager([{ name: 'spy' }], { spy: Spy });
    await mgr.onReset(NO_MODEL);
    await mgr.onReset(NO_MODEL);
    expect(calls).toBe(2);
  });

  it('warns and skips an unknown registry name', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const mgr = new EventManager([{ name: 'nope' }], {});
    expect(mgr.size).toBe(0);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

describe('EventManager: mode="reset" ONNX terms', () => {
  it('fires on reset, gated by size/session', async () => {
    const runFn = vi.fn(() => ({}));
    const deps = await depsWithSession('event/reset_x.onnx', fakeSession(runFn));
    const mgr = new EventManager(
      [{ name: 'reset_x', mode: 'reset', onnx: 'event/reset_x.onnx', rand_dim: 2 }],
      {},
      deps,
    );
    expect(mgr.size).toBe(1);
    await mgr.onReset(NO_MODEL);
    expect(runFn).toHaveBeenCalledTimes(1);
  });

  it('skips a config missing a loaded session (warns, does not throw)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const deps: EventManagerDeps = { sessions: new OnnxSessionCache(), rng: new SeededRng(1) };
    const mgr = new EventManager(
      [{ name: 'reset_x', mode: 'reset', onnx: 'missing.onnx', rand_dim: 2 }],
      {},
      deps,
    );
    expect(mgr.size).toBe(0);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('min_step_count_between_reset suppresses a too-soon reset', async () => {
    const runFn = vi.fn(() => ({}));
    const deps = await depsWithSession('event/x.onnx', fakeSession(runFn));
    const mgr = new EventManager(
      [
        {
          name: 'x',
          mode: 'reset',
          onnx: 'event/x.onnx',
          rand_dim: 1,
          min_step_count_between_reset: 5,
        },
      ],
      {},
      deps,
    );
    await mgr.onReset(NO_MODEL); // first reset always allowed
    await mgr.onReset(NO_MODEL); // immediately after -> gated
    expect(runFn).toHaveBeenCalledTimes(1);
    for (let i = 0; i < 5; i++) mgr.tick(0.02, NO_MODEL);
    await mgr.onReset(NO_MODEL); // enough steps have passed
    expect(runFn).toHaveBeenCalledTimes(2);
  });
});

describe('EventManager: mode="interval" — genuinely new dispatch', () => {
  it('does not call ort.run() on a quiet frame (fusion changes graph count, not call frequency)', async () => {
    const runFn = vi.fn(() => ({}));
    const deps = await depsWithSession('event/push_robot.onnx', fakeSession(runFn));
    const mgr = new EventManager(
      [
        {
          name: 'push_robot',
          mode: 'interval',
          onnx: 'event/push_robot.onnx',
          rand_dim: 6,
          interval_range_s: [1.0, 1.0],
        },
      ],
      {},
      deps,
    );
    for (let i = 0; i < 9; i++) {
      mgr.tick(0.1, NO_MODEL);
      await settle();
    }
    expect(runFn).not.toHaveBeenCalled();
  });

  it('fires once the interval elapses', async () => {
    const runFn = vi.fn(() => ({}));
    const deps = await depsWithSession('event/push_robot.onnx', fakeSession(runFn));
    const mgr = new EventManager(
      [
        {
          name: 'push_robot',
          mode: 'interval',
          onnx: 'event/push_robot.onnx',
          rand_dim: 6,
          interval_range_s: [1.0, 1.0],
        },
      ],
      {},
      deps,
    );
    for (let i = 0; i < 12; i++) {
      mgr.tick(0.25, NO_MODEL);
      await settle();
    }
    expect(runFn).toHaveBeenCalledTimes(3);
  });
});

describe('EventManager: mode="startup"', () => {
  it('fires exactly once, on startup(), not on reset or tick', async () => {
    const runFn = vi.fn(() => ({}));
    const deps = await depsWithSession('event/foot_friction.onnx', fakeSession(runFn));
    const mgr = new EventManager(
      [{ name: 'foot_friction', mode: 'startup', onnx: 'event/foot_friction.onnx', rand_dim: 1 }],
      {},
      deps,
    );
    await mgr.startup(NO_MODEL);
    await mgr.startup(NO_MODEL); // idempotent — StartupTrigger fires once
    await mgr.onReset(NO_MODEL);
    mgr.tick(10, NO_MODEL);
    expect(runFn).toHaveBeenCalledTimes(1);
  });
});
