/**
 * `FusedObservation`: one graph for a whole observation group (ADR 0005 §4).
 *
 * The per-term `OnnxObservation` is still how a group that cannot fuse runs, but
 * where a group *can* fuse this replaces all of its terms with a single
 * `ort.run()`. The measured motivation is in the companion brief §4b: a per-term
 * graph is often one node — three of Velocity-Flat-G1's five are `Identity`,
 * because the term body is just `sensor.data` — so the fixed per-call cost (the
 * JS↔WASM crossing, tensor marshalling, a promise round-trip) is the entire
 * expense, and a slot two terms share gets read and marshalled twice.
 *
 * What the build folded into the graph, so this class does not repeat it:
 * per-term `clip` then `scale` (mjlab's order), and the concatenation itself. The
 * output *is* the group vector.
 *
 * Two input kinds. Declared slots come from the `SlotReader` like any traced term.
 * Native terms — `prev_action` and a generated command — are graph inputs rather
 * than bodies: they read env-level state the runtime already holds, so feeding
 * them in keeps the output complete instead of something the caller must splice
 * offsets into. A `command` input's name is bound at construction
 * (`assertCommandTermBound`), because an unbound one would otherwise arrive as a
 * zero block inside the policy's input vector.
 *
 * History stays out. It is state across frames, which a stateless graph cannot
 * hold, and the build refuses to fuse a group whose terms carry their own
 * `history_length` (mjlab stacks per term *before* concatenating, so a group-level
 * ring buffer over the fused vector would order it differently).
 */

import { ObservationBase, type ObservationConfig } from './ObservationBase';
import { assertCommandTermBound, sliceStoredActions } from './NativeObservation';
import { conformToSize } from './pipeline';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import { slotDims, slotInputName } from '../onnx/session';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

/** A native term the fused graph takes as an input rather than computing. */
export interface FusedNativeInput {
  name: string;
  native: 'prev_action' | 'command';
  /** Graph input name to feed the value as. */
  input: string;
  size: number;
  /** `command` only: which command term to read. */
  command_name?: string;
  /** `prev_action` only: which action term, when the term names one. */
  action_name?: string;
  /** `prev_action` with an `action_name`: where that term's slice starts. */
  action_offset?: number;
}

export interface FusedObservationConfig extends ObservationConfig {
  /** Path to the group's graph (the field's presence is what marks a group fused). */
  fused: string;
  size: number;
  input_slots?: OnnxInputSlot[];
  native_inputs?: FusedNativeInput[];
  /** Per-term `{name, size}` in concat order, for the runner's group layout. */
  layout?: Array<{ name: string; size: number }>;
}

export interface FusedObservationDeps {
  session: OnnxSession;
  readSlot: SlotReader;
}

/** Whether a group config is a single fused graph rather than a list of terms. */
export function isFusedObservationConfig(entry: unknown): entry is FusedObservationConfig {
  return (
    typeof entry === 'object' &&
    entry !== null &&
    typeof (entry as { fused?: unknown }).fused === 'string'
  );
}

export class FusedObservation extends ObservationBase<FusedObservationConfig> {
  private readonly deps: FusedObservationDeps;
  /** Last completed vector, served if a later frame's inference cannot run. */
  private last: Float32Array;

  constructor(
    runner: PolicyRunner,
    config: FusedObservationConfig,
    deps: FusedObservationDeps,
  ) {
    super(runner, config);
    for (const native of config.native_inputs ?? []) {
      if (native.native === 'command') {
        assertCommandTermBound(runner, `${config.name}.${native.name}`, native.command_name);
      }
    }
    this.deps = deps;
    this.last = new Float32Array(config.size);
  }

  get size(): number {
    return this.config.size;
  }

  async compute(_state: PolicyState): Promise<Float32Array> {
    const feeds: Record<string, OnnxTensorLike> = {};
    for (const slot of this.config.input_slots ?? []) {
      const value = this.deps.readSlot(slot);
      if (!value) {
        // Serve the last good vector rather than feeding the policy zeros for a
        // slice of its input — same call as the per-term handler makes.
        console.warn(
          `[FusedObservation] "${this.config.name}" could not read slot ` +
            `${slotInputName(slot)}; reusing the previous vector.`,
        );
        return this.last;
      }
      feeds[slotInputName(slot)] = { data: value, dims: slotDims(slot, value.length) };
    }
    for (const native of this.config.native_inputs ?? []) {
      const value = this.readNative(native);
      feeds[native.input] = { data: value, dims: [1, value.length] };
    }

    const outputs = await this.deps.session.run(feeds);
    const first = Object.values(outputs)[0];
    if (!first) {
      console.warn(`[FusedObservation] "${this.config.name}" produced no output.`);
      return this.last;
    }
    this.last = conformToSize(Float32Array.from(first.data as Float32Array), this.size);
    return this.last;
  }

  private readNative(native: FusedNativeInput): Float32Array {
    const raw =
      native.native === 'prev_action'
        ? sliceStoredActions(this.runner.getLastActions(), native)
        : this.readCommand(native);
    // Copied and conformed: `raw` may be the runtime's own buffer, and the graph
    // declared a fixed width the feed has to match.
    return conformToSize(Float32Array.from(raw), native.size);
  }

  /**
   * The named command's current vector.
   *
   * The name and its presence in the manager are both asserted in the constructor,
   * so the only fallback left is an embedding with no `CommandManager` at all, whose
   * width comes from the build. This used to end in `?? new Float32Array(size)`,
   * which never fired — `getCommand` returns an empty (truthy) `Float32Array` for a
   * miss, not `null`, so the zero-fill happened a line later in `conformToSize` and
   * the coalesce read as a guard while guarding nothing.
   */
  private readCommand(native: FusedNativeInput): Float32Array {
    const manager = this.runner.getContext()?.commandManager;
    if (!manager || !native.command_name) return new Float32Array(native.size);
    return manager.getCommand(native.command_name);
  }
}
