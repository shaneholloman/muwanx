/**
 * The reset chain's ordering, as one awaited sequence (ADR 0005 §8).
 *
 * mjlab runs `event_manager.apply(mode="reset")` **to completion**, then
 * `command_manager.reset` (`envs/manager_based_rl_env.py`, `_reset_idx`). That order
 * is load-bearing rather than cosmetic, because both sides write `qpos`: a reset
 * event writes the perturbed initial state (`reset_joints_by_offset`,
 * `reset_root_state_uniform`), and `TrackingCommand.reset` writes the reference pose
 * through `applyReferenceStateToSim`. mjlab's writes are assignments in that order,
 * so where they overlap the command's value is the one that survives.
 *
 * `runtime.ts` used to fire the events un-awaited (`void eventManager.onReset(…)`)
 * and reset the command terms on the very next line. That put the *command's*
 * synchronous write first and the event's ORT-backed write a frame or two later —
 * the same two writes, resolved the other way round from mjlab. Awaiting restores
 * the order, and also stops a second reset arriving mid-inference from being dropped
 * by `OnnxEvent`'s own in-flight guard.
 *
 * Extracted from `runtime.ts` rather than inlined there because `runtime.ts` needs
 * the MuJoCo WASM module to instantiate, so nothing in it is unit-testable — and an
 * ordering guarantee with no test is one `void` away from silently reverting.
 * Structural parameter types for the same reason: the contract is "await the events,
 * then reset the commands", not the two concrete managers.
 */

import type { EventContext } from '../event/EventBase';

/** The reset-mode event dispatch this chain awaits — `EventManager`. */
export interface ResetEventSource {
  onReset(context: EventContext): Promise<void>;
}

/** The command terms reset once the events have landed — `CommandManager`. */
export interface ResetCommandSink {
  resetTerms(): void;
}

/**
 * Fire `mode="reset"` events, wait for them, then reset the command terms.
 *
 * A scene with no event manager still resets its commands: the ordering constraint
 * is vacuous there, not a reason to skip the second half.
 */
export async function applyResetTerms(
  events: ResetEventSource | null | undefined,
  commands: ResetCommandSink,
  context: EventContext,
): Promise<void> {
  await events?.onReset(context);
  commands.resetTerms();
}
