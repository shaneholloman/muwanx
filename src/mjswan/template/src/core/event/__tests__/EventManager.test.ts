/**
 * Mode-aware event dispatch: interval/startup dispatch, reset staying compatible with
 * plugin-registered term classes, and a quiet frame never calling `ort.run()` for a
 * term whose trigger did not fire.
 */
import { describe, expect, it, vi } from 'vitest';

import { SeededRng } from '../../rng';
import { OnnxSessionCache, type OnnxSession, type OnnxTensorLike } from '../../onnx/session';
import { EventManager, type EventManagerDeps } from '../EventManager';
import { EventBase, type EventContext } from '../EventBase';
import type { ModelFieldDrConfig } from '../modelFieldDr';

const NO_MODEL: EventContext = { mjModel: null, mjData: null };

/** Drain the microtask queue, so an interval term's in-flight guard clears between ticks. */
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

  it('counts a model-field randomization as a loaded term', () => {
    // A task whose only events are DR would otherwise report "0 loaded" while randomizing.
    const dr: ModelFieldDrConfig = {
      name: 'base_com',
      kind: 'model_field',
      field: 'body_ipos',
      entity_type: 'body',
      entity_names: ['robot/torso_link'],
      axis_ranges: { 0: [-0.025, 0.025] },
      operation: 'add',
      distribution: 'uniform',
      shared_random: false,
      uses_defaults: true,
      set_const: true,
    };
    expect(new EventManager([dr], {}).size).toBe(1);
  });

  it('skips a build-marked native term without warning', () => {
    // The build's `reason`, not a missing plugin the reader would go looking for.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const mgr = new EventManager(
      [{ name: 'encoder_bias', mode: 'startup', native: true, reason: 'wrote nothing traceable' }],
      {},
    );
    expect(mgr.size).toBe(0);
    expect(warn).not.toHaveBeenCalled();
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

describe('EventManager: mode="manual" — the operator is the schedule', () => {
  /** One manual term and one interval term over the same graph path. */
  async function bothKinds(runFn: () => Record<string, never>) {
    const deps = await depsWithSession('event/throw.onnx', fakeSession(runFn));
    return new EventManager(
      [
        {
          name: 'throw_overhead',
          mode: 'manual',
          onnx: 'event/throw.onnx',
          rand_dim: 0,
          label: 'Throw overhead',
        },
        {
          name: 'throw_ball',
          mode: 'interval',
          onnx: 'event/throw.onnx',
          rand_dim: 0,
          interval_range_s: [1.0, 1.0],
          label: 'Auto throw',
        },
      ],
      {},
      deps,
    );
  }

  it('never fires on its own — not on startup, reset or a tick', async () => {
    const runFn = vi.fn(() => ({}));
    const deps = await depsWithSession('event/throw.onnx', fakeSession(runFn));
    const mgr = new EventManager(
      [{ name: 'throw_overhead', mode: 'manual', onnx: 'event/throw.onnx', rand_dim: 0 }],
      {},
      deps,
    );
    await mgr.startup(NO_MODEL);
    await mgr.onReset(NO_MODEL);
    await mgr.tick(10, NO_MODEL);
    expect(runFn).not.toHaveBeenCalled();
    // Counted all the same: a scene whose only event is manual is not "0 loaded".
    expect(mgr.size).toBe(1);
  });

  it('fires when asked, by name, as often as asked', async () => {
    const runFn = vi.fn(() => ({}));
    const mgr = await bothKinds(runFn);
    await mgr.fire('throw_overhead', NO_MODEL);
    await mgr.fire('throw_overhead', NO_MODEL);
    expect(runFn).toHaveBeenCalledTimes(2);
  });

  it('warns rather than throwing for a name that is not a manual term', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const runFn = vi.fn(() => ({}));
    const mgr = await bothKinds(runFn);
    // An interval term is not the operator's to fire directly.
    await mgr.fire('throw_ball', NO_MODEL);
    expect(runFn).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('offers one control per term, labelled as the build declared', async () => {
    const mgr = await bothKinds(vi.fn(() => ({})));
    expect(mgr.controls()).toEqual([
      { name: 'throw_overhead', label: 'Throw overhead', kind: 'manual', armed: true },
      { name: 'throw_ball', label: 'Auto throw', kind: 'interval', armed: true },
    ]);
  });

  it('falls back to the term name when the build declared no label', async () => {
    const deps = await depsWithSession('event/throw.onnx', fakeSession(vi.fn(() => ({}))));
    const mgr = new EventManager(
      [{ name: 'throw_overhead', mode: 'manual', onnx: 'event/throw.onnx', rand_dim: 0 }],
      {},
      deps,
    );
    expect(mgr.controls()[0].label).toBe('throw_overhead');
  });

  it('disarming an interval term stops it while the button keeps working', async () => {
    const runFn = vi.fn(() => ({}));
    const mgr = await bothKinds(runFn);
    expect(mgr.setArmed('throw_ball', false)).toBe(true);
    for (let i = 0; i < 12; i++) {
      await mgr.tick(0.25, NO_MODEL);
      await settle();
    }
    expect(runFn).not.toHaveBeenCalled();
    expect(mgr.controls()[1].armed).toBe(false);

    // The operator's own throw is unaffected by the schedule being off.
    await mgr.fire('throw_overhead', NO_MODEL);
    expect(runFn).toHaveBeenCalledTimes(1);

    // Re-armed, it runs again — after a fresh interval, not the moment it comes back.
    expect(mgr.setArmed('throw_ball', true)).toBe(true);
    await mgr.tick(0.25, NO_MODEL);
    await settle();
    expect(runFn).toHaveBeenCalledTimes(1);
    for (let i = 0; i < 4; i++) {
      await mgr.tick(0.25, NO_MODEL);
      await settle();
    }
    expect(runFn).toHaveBeenCalledTimes(2);
  });

  it('disarms the button while its `disabled_when` schedule is armed', async () => {
    const runFn = vi.fn(() => ({}));
    const deps = await depsWithSession('event/throw.onnx', fakeSession(runFn));
    const mgr = new EventManager(
      [
        {
          name: 'throw_overhead',
          mode: 'manual',
          onnx: 'event/throw.onnx',
          rand_dim: 0,
          disabled_when: 'throw_ball',
        },
        {
          name: 'throw_ball',
          mode: 'interval',
          onnx: 'event/throw.onnx',
          rand_dim: 0,
          interval_range_s: [1.0, 1.0],
        },
      ],
      {},
      deps,
    );
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // Interval terms load armed, so the panel greys the button out from the start.
    expect(mgr.controls()[0].armed).toBe(false);
    await mgr.fire('throw_overhead', NO_MODEL);
    expect(runFn).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();

    // Auto throw off: the button is the only thrower again.
    expect(mgr.setArmed('throw_ball', false)).toBe(true);
    expect(mgr.controls()[0].armed).toBe(true);
    await mgr.fire('throw_overhead', NO_MODEL);
    expect(runFn).toHaveBeenCalledTimes(1);
  });

  it('reports a name it cannot arm rather than pretending', async () => {
    const mgr = await bothKinds(vi.fn(() => ({})));
    expect(mgr.setArmed('throw_overhead', false)).toBe(false);
    expect(mgr.setArmed('nope', false)).toBe(false);
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

// ---------------------------------------------------------------------------
// Ordering. mjlab loops a mode's terms in config order and every write is an
// assignment, so two terms touching one element resolve last-writer-wins by config
// order — not by whichever graph resolved first.
// ---------------------------------------------------------------------------

describe('EventManager: dispatch order matches mjlab', () => {
  /** Two reset terms whose sessions resolve out of config order. */
  function outOfOrderDeps(applied: string[]): Promise<EventManagerDeps> {
    // `slow` is declared first but resolves last, so `Promise.all` would invert them.
    const slow: OnnxSession = {
      run: async () => {
        for (let i = 0; i < 8; i++) await Promise.resolve();
        applied.push('slow');
        return { next_dummy: { data: Float32Array.from([0]), dims: [1, 1] } };
      },
    };
    const fast: OnnxSession = {
      run: () => {
        applied.push('fast');
        return Promise.resolve({ next_dummy: { data: Float32Array.from([0]), dims: [1, 1] } });
      },
    };
    const sessions = new OnnxSessionCache((data) =>
      Promise.resolve(new Uint8Array(data)[0] === 1 ? slow : fast),
    );
    return sessions
      .load([
        { name: 'event/slow.onnx', data: Uint8Array.from([1]).buffer },
        { name: 'event/fast.onnx', data: Uint8Array.from([2]).buffer },
      ])
      .then(() => ({ sessions, rng: new SeededRng(1) }));
  }

  it('fires reset terms in config order, not completion order', async () => {
    const applied: string[] = [];
    const manager = new EventManager(
      [
        { name: 'slow', mode: 'reset', onnx: 'event/slow.onnx', rand_dim: 0 },
        { name: 'fast', mode: 'reset', onnx: 'event/fast.onnx', rand_dim: 0 },
      ] as never,
      {},
      await outOfOrderDeps(applied),
    );
    await manager.onReset(NO_MODEL);
    // Config order; `Promise.all` would give ['fast', 'slow'] and let `slow` win.
    expect(applied).toEqual(['slow', 'fast']);
  });

  it('fires startup terms in config order too', async () => {
    const applied: string[] = [];
    const manager = new EventManager(
      [
        { name: 'slow', mode: 'startup', onnx: 'event/slow.onnx', rand_dim: 0 },
        { name: 'fast', mode: 'startup', onnx: 'event/fast.onnx', rand_dim: 0 },
      ] as never,
      {},
      await outOfOrderDeps(applied),
    );
    await manager.startup(NO_MODEL);
    expect(applied).toEqual(['slow', 'fast']);
  });

  it('fires interval terms in config order too', async () => {
    // mjlab runs one loop — decrement, fire, next — so config order decides an overlap.
    const applied: string[] = [];
    const manager = new EventManager(
      [
        {
          name: 'slow',
          mode: 'interval',
          onnx: 'event/slow.onnx',
          rand_dim: 0,
          interval_range_s: [0.01, 0.01],
        },
        {
          name: 'fast',
          mode: 'interval',
          onnx: 'event/fast.onnx',
          rand_dim: 0,
          interval_range_s: [0.01, 0.01],
        },
      ] as never,
      {},
      await outOfOrderDeps(applied),
    );
    // One dt past both timers, so both fire on this tick.
    await manager.tick(0.02, NO_MODEL);
    expect(applied).toEqual(['slow', 'fast']);
  });

  it('advances the reset gates even when an interval term fails', async () => {
    // The gate counters advance before anything can throw: a bad graph must not stall them.
    const sessions = new OnnxSessionCache(() =>
      Promise.resolve({ run: () => Promise.reject(new Error('graph missing')) }),
    );
    await sessions.load([{ name: 'event/boom.onnx', data: new ArrayBuffer(1) }]);
    const manager = new EventManager(
      [
        {
          name: 'boom',
          mode: 'interval',
          onnx: 'event/boom.onnx',
          rand_dim: 0,
          interval_range_s: [0.01, 0.01],
        },
        { name: 'gated', mode: 'reset', onnx: 'event/boom.onnx', rand_dim: 0 },
      ] as never,
      {},
      { sessions, rng: new SeededRng(1) },
    );
    // `OnnxEvent.fire` reports a missing model, so this rejects only via the session.
    await expect(manager.tick(0.02, NO_MODEL)).rejects.toThrow('graph missing');
  });
});
