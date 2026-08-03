/**
 * The reset chain's ordering, as one awaited sequence (ADR 0005 §8).
 *
 * mjlab's `_reset_idx` (`envs/manager_based_rl_env.py`) is explicitly order
 * sensitive. Of the managers mjswan keeps, its order is:
 *
 *     event(mode="reset")  →  observation.reset  →  action.reset  →  command.reset
 *
 * followed by `termination.reset`, which the caller does after this returns.
 *
 * **Events before commands** is load-bearing rather than cosmetic, because both
 * sides write `qpos`: a reset event writes the perturbed initial state
 * (`reset_joints_by_offset`, `reset_root_state_uniform`), and
 * `TrackingCommand.reset` writes the reference pose through
 * `applyReferenceStateToSim`. mjlab's writes are assignments in the order above, so
 * where they overlap the command's value is the one that survives. `runtime.ts` used
 * to fire the events un-awaited (`void eventManager.onReset(…)`) and reset the
 * command terms on the very next line — which put the *command's* synchronous write
 * first and the event's ORT-backed write a frame or two later, the same two writes
 * resolved the other way round. Awaiting also stops a second reset arriving
 * mid-inference from being dropped by `OnnxEvent`'s own in-flight guard.
 *
 * **Observation and action before commands** has no such consequence today: no
 * reference task has a command term that reads `last_action` or an observation. It
 * is matched anyway because the inversion had no reason behind it — `runtime.ts` ran
 * `policyRunner.reset` after the command reset purely because that was where the
 * call happened to sit — and a divergence kept for no reason is one a future command
 * term silently inherits.
 *
 * Extracted from `runtime.ts` rather than inlined there because `runtime.ts` needs
 * the MuJoCo WASM module to instantiate, so nothing in it is unit-testable — and an
 * ordering guarantee with no test is one `void` away from silently reverting.
 * Structural parameter types for the same reason: the contract is the order, not the
 * three concrete managers.
 */

import type { EventContext } from '../event/EventBase';
import type { PolicyState } from '../policy/types';

/** The reset-mode event dispatch this chain awaits — `EventManager`. */
export interface ResetEventSource {
  onReset(context: EventContext): Promise<void>;
}

/**
 * mjlab's `observation_manager.reset` + `action_manager.reset` in one call —
 * `PolicyRunner.reset`, which clears the observation history and zeroes the stored
 * actions.
 */
export interface ResetPolicySink {
  reset(state?: PolicyState): void;
}

/**
 * The command terms reset last — `CommandManager`.
 *
 * Awaited, because a traced term's reset *is* its resample: mjlab's
 * `CommandTerm.reset` calls `_resample(env_ids)`, which for `OnnxCommand` is an
 * `ort.run()` that may write to the sim. Those writes have to land before the
 * caller's forward, which is what publishes them.
 */
export interface ResetCommandSink {
  resetTerms(): void | Promise<void>;
}

export interface ResetChain {
  events?: ResetEventSource | null;
  policy?: ResetPolicySink | null;
  commands: ResetCommandSink;
  context: EventContext;
  /**
   * The state handed to the policy reset.
   *
   * A thunk, not a value: it has to be read *after* the reset events' writes land,
   * so a caller running before the await cannot supply it.
   */
  buildState?: () => PolicyState | undefined;
}

/**
 * Fire `mode="reset"` events, wait for them, then reset the policy and the command
 * terms — mjlab's `_reset_idx` order.
 *
 * Absent managers are skipped rather than treated as errors: a scene with no events
 * still resets its commands (the ordering constraint is vacuous there, not a reason
 * to drop the rest), and `loadPolicyConfig` runs this before a `PolicyRunner`
 * exists.
 */
export async function applyResetTerms(chain: ResetChain): Promise<void> {
  await chain.events?.onReset(chain.context);
  chain.policy?.reset(chain.buildState?.());
  await chain.commands.resetTerms();
}
