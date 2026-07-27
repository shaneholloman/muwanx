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

export class NativeObservation extends ObservationBase<NativeObservationConfig> {
  private readonly constant: Float32Array | null;
  private cachedSize: number | null;

  constructor(runner: PolicyRunner, config: NativeObservationConfig) {
    super(runner, config);
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
        return this.runner.getLastActions();
      case 'command': {
        const name = this.config.command_name;
        const manager = this.runner.getContext()?.commandManager;
        if (!name || !manager) return new Float32Array(this.cachedSize ?? 0);
        return manager.getCommand(name);
      }
    }
  }
}
