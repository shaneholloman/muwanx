/**
 * mjlab's order-sensitive `_reset_idx`, as one awaited sequence:
 *
 *     event(mode="reset")  →  observation.reset  →  action.reset  →  command.reset
 *
 * then `termination.reset`, which the caller does after this returns.
 *
 * **Events before commands** is load-bearing: both write `qpos`, and mjlab's order
 * decides which survives. Awaiting also keeps a second reset from being dropped by
 * `OnnxEvent`'s in-flight guard. The rest is matched for future terms' sake.
 *
 * A free function, not inline in `runtime.ts` (which needs the MuJoCo WASM to
 * instantiate), so the ordering has a test. Structural parameter types for the same
 * reason.
 */

import type { EventContext } from '../event/EventBase';
import type { PolicyState } from '../policy/types';

/** The reset-mode event dispatch this chain awaits — `EventManager`. */
export interface ResetEventSource {
  onReset(context: EventContext): Promise<void>;
}

/** `PolicyRunner.reset`: mjlab's observation + action manager resets in one call. */
export interface ResetPolicySink {
  reset(state?: PolicyState): void;
}

/** The command terms, reset last; awaited so a resample's writes precede the forward. */
export interface ResetCommandSink {
  resetTerms(): void | Promise<void>;
}

export interface ResetChain {
  events?: ResetEventSource | null;
  policy?: ResetPolicySink | null;
  commands: ResetCommandSink;
  context: EventContext;
  /** A thunk, since it has to be read after the reset events' writes land. */
  buildState?: () => PolicyState | undefined;
}

/**
 * Fire `mode="reset"` events, wait for them, then reset the policy and the command terms.
 * Absent managers are skipped — `loadPolicyConfig` runs this before a `PolicyRunner` exists.
 */
export async function applyResetTerms(chain: ResetChain): Promise<void> {
  await chain.events?.onReset(chain.context);
  chain.policy?.reset(chain.buildState?.());
  await chain.commands.resetTerms();
}
