/**
 * `NativeObservation`: the observation terms that are a plain read of state the
 * orchestrator already owns, so tracing them to ONNX would only wrap an identity
 * graph around a value the runtime has in hand (ADR 0005 §Decision — orchestration
 * is native, term *bodies* are ONNX; these terms have no body).
 *
 * The build marks them with `native`:
 * - `prev_action` — mjlab's `last_action`, i.e. the policy's own previous output.
 * - `command` — mjlab's `generated_commands`, the named command term's current
 *   value (itself possibly produced by an `OnnxCommand`).
 * - `constant` — a term that reads nothing from the env at all, e.g. zero padding
 *   for command slots a checkpoint expects but this scene does not drive. Its
 *   value is baked at build time.
 *
 * `size` is normally supplied by the build; for `prev_action`/`command` it can be
 * resolved from the runtime instead, since a scene may name a command that only
 * exists browser-side (a native `UiCommand`) and so has no build-time width.
 */

import { ObservationBase, type ObservationConfig } from './ObservationBase';
import {
  applyObservationPipeline,
  conformToSize,
  type ObservationClip,
  type ObservationScale,
} from './pipeline';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

export type NativeObservationKind = 'prev_action' | 'command' | 'constant';

export interface NativeObservationConfig extends ObservationConfig {
  native: NativeObservationKind;
  size?: number;
  /** `constant` only: the baked value. */
  value?: number[];
  /** `command` only: which command term to read. */
  command_name?: string;
  /** `prev_action` only: which action term, when the term names one. */
  action_name?: string;
  /** `prev_action` with an `action_name`: where that term's slice starts. */
  action_offset?: number;
  scale?: ObservationScale;
  clip?: ObservationClip;
}

/** Whether a config entry names a natively-computed observation. */
export function isNativeObservationConfig(
  entry: ObservationConfig,
): entry is NativeObservationConfig {
  const native = (entry as { native?: unknown }).native;
  return native === 'prev_action' || native === 'command' || native === 'constant';
}

/**
 * The stored actions, narrowed to one action term's slice.
 *
 * `last_action(action_name=…)` is mjlab's `get_term(name).raw_action` — that term's
 * slice of the policy output, not the whole vector (`envs/mdp/observations.py`). The
 * build resolves where the slice starts from the live `ActionManager` and emits it as
 * `action_offset`; an entry without one is the bare `last_action`, which *is* the
 * whole vector.
 *
 * The build emitted `action_name` from the start and nothing here read it, so a
 * named term got the vector's head instead of its own slice — `conformToSize`
 * truncates from the front, which made the wrong numbers the right width. Invisible
 * while every reference task has exactly one action term (the slice and the vector
 * coincide), wrong the moment one has two.
 */
export function sliceStoredActions(
  actions: Float32Array,
  config: { action_offset?: number; size?: number },
): Float32Array {
  const offset = config.action_offset;
  if (offset === undefined) return actions;
  // A view, not a copy: both callers copy before the pipeline mutates, and
  // `getLastActions()` already hands back a copy of the runtime's buffer.
  return actions.subarray(offset, offset + (config.size ?? actions.length - offset));
}

/**
 * Fail at construction if a `command` term names something no command term provides.
 *
 * mjlab asserts this lookup (`envs/mdp/observations.py`, `generated_commands`).
 * Here the miss used to surface as `CommandManager.getCommand`'s empty vector,
 * zero-padded to the declared width by `conformToSize` — a block of zeros inside
 * the policy's input vector, warned about nowhere. A scene that deliberately has no
 * value for a command slot already has a way to say so (`native: "constant"`, whose
 * value the build bakes), so a dangling name is always a wiring bug.
 *
 * Lives here rather than in either observation class because both bind these names:
 * this module owns the `native` kinds, and `FusedObservation` feeds the same two as
 * graph inputs.
 *
 * Only checked when a manager is present. An embedding that runs no commands at all
 * is not evidence of a wrong *name*, so the declared width still stands zero-filled
 * for it — the same call the width-conforming path has always made.
 */
export function assertCommandTermBound(
  runner: PolicyRunner,
  label: string,
  commandName: string | undefined,
): void {
  if (!commandName) {
    throw new Error(
      `Observation term "${label}" is native:"command" but carries no command_name, ` +
        'so there is nothing to read. The build always emits one (mjlab takes it as a ' +
        'required param), which makes this a malformed config rather than a scene ' +
        'without commands.',
    );
  }
  const manager = runner.getContext()?.commandManager;
  if (!manager) return;
  const available = manager.termNames();
  if (available.includes(commandName)) return;
  throw new Error(
    `Observation term "${label}" reads the command "${commandName}", which this scene ` +
      `does not define. Available: ${available.length ? available.join(', ') : '(none)'}. ` +
      'Unchecked, this feeds the policy a zero block of the declared width — a ' +
      'silently wrong input vector — so the scene fails to load instead. A command ' +
      'slot the scene deliberately does not drive belongs as native:"constant".',
  );
}

export class NativeObservation extends ObservationBase<NativeObservationConfig> {
  private readonly constant: Float32Array | null;
  private cachedSize: number | null;

  constructor(runner: PolicyRunner, config: NativeObservationConfig) {
    super(runner, config);
    if (config.native === 'command') {
      assertCommandTermBound(runner, config.name, config.command_name);
    }
    this.constant =
      config.native === 'constant' ? Float32Array.from(config.value ?? []) : null;
    this.cachedSize = config.size ?? this.constant?.length ?? null;
  }

  get size(): number {
    if (this.cachedSize !== null) return this.cachedSize;
    // No build-time width (a browser-only command): take it from the live value
    // once, so the group layout stays fixed from then on.
    this.cachedSize = this.read().length;
    return this.cachedSize;
  }

  compute(_state: PolicyState): Float32Array {
    const raw = this.read();
    // Copy before the pipeline mutates in place — `read()` may hand back the
    // runtime's own buffer (the last action, a command's vector).
    const values = conformToSize(Float32Array.from(raw), this.size);
    return applyObservationPipeline(values, this.config);
  }

  private read(): Float32Array {
    switch (this.config.native) {
      case 'constant':
        return this.constant ?? new Float32Array(0);
      case 'prev_action':
        return sliceStoredActions(this.runner.getLastActions(), this.config);
      case 'command': {
        const name = this.config.command_name;
        const manager = this.runner.getContext()?.commandManager;
        // `name` and its presence in the manager are both asserted in the
        // constructor, so the only case this guard still catches is an embedding
        // with no CommandManager — whose width the build supplied.
        if (!name || !manager) return new Float32Array(this.cachedSize ?? 0);
        return manager.getCommand(name);
      }
    }
  }
}
