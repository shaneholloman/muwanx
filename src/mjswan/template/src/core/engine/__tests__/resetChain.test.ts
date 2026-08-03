/**
 * The reset chain's ordering (ADR 0005 §8, mjlab's `_reset_idx`).
 *
 * `runtime.ts` needs the MuJoCo WASM module to instantiate, so the order it used to
 * express inline — `void eventManager.onReset(…)` followed immediately by
 * `commandManager.resetTerms()` — had no test and no way to get one. These cover the
 * extracted sequence, and every one of them passes under `void` except the ordering
 * ones, which is the point: they fail the moment the await is dropped again.
 */
import { describe, expect, it } from 'vitest';

import { applyResetTerms } from '../resetChain';
import type { EventContext } from '../../event/EventBase';

/** The chain only forwards this, so an empty scene context is enough. */
const CONTEXT = {
  mujoco: null,
  mjModel: null,
  mjData: null,
  terrainData: null,
} as unknown as EventContext;

/** Yield past the microtask queue, the way an awaited ORT run does. */
function settle(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

describe('applyResetTerms', () => {
  it('resets the command terms only after the reset events have landed', async () => {
    const order: string[] = [];
    const events = {
      onReset: async () => {
        await settle();
        order.push('event');
      },
    };
    const commands = { resetTerms: () => order.push('command') };

    await applyResetTerms(events, commands, CONTEXT);

    // Un-awaited (`void events.onReset()`) gives ['command', 'event'] — mjlab runs
    // `event_manager.apply(mode="reset")` to completion first.
    expect(order).toEqual(['event', 'command']);
  });

  it('lets the command term win where both write the same element', async () => {
    // The concrete bug, in miniature. A reset event writes `qpos` (ORT-backed, so
    // async) and `TrackingCommand.reset` writes `qpos` too (synchronous). mjlab
    // assigns in `_reset_idx` order, so the command's value is the survivor.
    const qpos = [0];
    const events = {
      onReset: async () => {
        await settle();
        qpos[0] = 1;
      },
    };
    const commands = { resetTerms: () => { qpos[0] = 2; } };

    await applyResetTerms(events, commands, CONTEXT);
    // Settle *after* the call as well: an un-awaited event write lands here rather
    // than before the assertion, so without this the test passes either way.
    await settle();

    // Un-awaited leaves 1 — the command writes first and the event overwrites it,
    // which is mjlab's order backwards.
    expect(qpos[0]).toBe(2);
  });

  it('still resets the command terms when the scene has no event manager', async () => {
    const order: string[] = [];
    await applyResetTerms(null, { resetTerms: () => order.push('command') }, CONTEXT);
    expect(order).toEqual(['command']);
  });

  it('propagates a failing reset event instead of resetting commands on top of it', async () => {
    // The caller decides what a failed reset means (the step loop lets it surface,
    // the sync public verb logs it). Swallowing it here would reset the commands
    // against a half-applied scene and report nothing.
    let reset = false;
    const events = { onReset: () => Promise.reject(new Error('graph missing')) };
    await expect(
      applyResetTerms(events, { resetTerms: () => { reset = true; } }, CONTEXT),
    ).rejects.toThrow('graph missing');
    expect(reset).toBe(false);
  });
});
