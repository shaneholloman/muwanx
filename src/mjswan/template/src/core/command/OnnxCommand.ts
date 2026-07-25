/**
 * `OnnxCommand`: the single generic command handler (ADR 0005 §3, brief §3).
 *
 * Every traced command — `UniformVelocityCommand`, `LiftingCommand`, … — is a
 * *data instantiation* of this one class: a different `.onnx` graph, different
 * `state_fields`, and different (or absent) `ui` / `write_targets` in
 * `policy.json`. There is deliberately no engine-side class per command.
 *
 * What this class owns (the native half; the graph owns the math):
 *
 * - **The resample timer** — a scalar countdown (ADR §5), not mjlab's per-env
 *   tensor. It sets `resample_mask` for the frame.
 * - **State across frames** — the graph is a pure function
 *   `(prev_state, resample_mask, rand, dynamic…) → (next_state, …)`; the handler
 *   holds `prev_state`, seeded from each field's declared shape/dtype.
 * - **`rand`** — drawn from the orchestrator-owned seeded PRNG so a session
 *   replays bit-for-bit (ADR §2). Never ONNX's own random ops.
 * - **The UI override** — compute autonomously *every* frame, then overwrite the
 *   command from UI values when the enable checkbox is on, matching mjlab's
 *   play-time behaviour exactly (brief §3a): the autonomous computation is never
 *   skipped.
 * - **`entity_write`** — hands graph-computed pose/velocity to the apply
 *   primitive (brief §3).
 *
 * **Async boundary.** `CommandTerm.update()`/`getCommand()` are synchronous, but
 * ORT-Web inference is not. `update()` therefore *kicks off* inference and
 * returns; `getCommand()` serves the most recently completed value. A frame that
 * arrives while inference is still in flight is skipped rather than queued, so
 * the command can never build a backlog. The resulting one-frame lag is the same
 * property already accepted elsewhere in the design (ADR §8).
 */

import { SeededRng } from '../rng';
import { applyEntityWrites, type WriteTarget, type WriteValues } from '../event/entityWrite';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import type { CommandConfigEntry, CommandTerm, CommandTermContext, CommandUiConfig } from './types';

export type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader };

export interface OnnxStateFieldSpec {
  name: string;
  shape: number[];
  dtype: string;
}

export interface OnnxCommandConfig extends CommandConfigEntry {
  onnx: string;
  command_field: string;
  rand_dim: number;
  state_fields: OnnxStateFieldSpec[];
  input_slots?: OnnxInputSlot[];
  write_targets?: WriteTarget[];
  resampling_time_range?: [number, number];
  rand_ranges?: Array<[number, number]>;
  debug_vis?: boolean;
}

export interface OnnxCommandDeps {
  session: OnnxSession;
  rng: SeededRng;
  readSlot?: SlotReader;
}

function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function makeTensor(spec: OnnxStateFieldSpec): OnnxTensorLike {
  const n = numel(spec.shape);
  const data = spec.dtype === 'bool' ? new Uint8Array(n) : new Float32Array(n);
  return { data, dims: [...spec.shape] };
}

export class OnnxCommand implements CommandTerm {
  private readonly cfg: OnnxCommandConfig;
  private readonly deps: OnnxCommandDeps;
  private readonly context: CommandTermContext | null;

  private state = new Map<string, OnnxTensorLike>();
  private command: Float32Array;
  private timeLeft = 0;
  private inFlight = false;
  /** Set on the first update so the initial frame resamples (reset semantics). */
  private pendingResample = true;
  private uiValues = new Map<string, number>();

  constructor(
    _termName: string,
    config: OnnxCommandConfig,
    context: CommandTermContext | null,
    deps: OnnxCommandDeps,
  ) {
    this.cfg = config;
    this.deps = deps;
    this.context = context;
    for (const spec of config.state_fields) this.state.set(spec.name, makeTensor(spec));
    const commandSpec = config.state_fields.find(s => s.name === config.command_field);
    this.command = new Float32Array(commandSpec ? numel(commandSpec.shape) : 0);
    this.timeLeft = this.sampleResampleTime();
    for (const input of config.ui?.inputs ?? []) {
      if (input.type === 'slider') this.uiValues.set(input.name, input.default);
      else if (input.type === 'checkbox') this.uiValues.set(input.name, input.default ? 1 : 0);
    }
  }

  getCommand(): Float32Array {
    // UI override: the autonomous value is already computed; overwrite per axis.
    if (this.isUiEnabled()) {
      const sliders = (this.cfg.ui?.inputs ?? []).filter(i => i.type === 'slider');
      const out = Float32Array.from(this.command);
      for (let i = 0; i < sliders.length && i < out.length; i++) {
        out[i] = this.uiValues.get(sliders[i].name) ?? out[i];
      }
      return out;
    }
    return this.command;
  }

  getUiConfig(): CommandUiConfig | null {
    return this.cfg.ui ?? null;
  }

  /** Advance the timer and kick off inference; never blocks (see class docs). */
  update(dt: number): void {
    this.timeLeft -= dt;
    if (this.timeLeft <= 0) {
      this.pendingResample = true;
      this.timeLeft += this.sampleResampleTime();
      if (this.timeLeft <= 0) this.timeLeft = this.sampleResampleTime();
    }
    if (this.inFlight) return; // skip, never queue
    const resample = this.pendingResample;
    this.pendingResample = false;
    this.inFlight = true;
    void this.step(resample).finally(() => {
      this.inFlight = false;
    });
  }

  /** Episode reset unifies to "resample next frame" (ADR §3). */
  reset(): void {
    this.pendingResample = true;
    this.timeLeft = this.sampleResampleTime();
  }

  setValue(inputName: string, value: number): number {
    this.uiValues.set(inputName, value);
    return value;
  }

  triggerButton(inputName: string): void {
    // A `zero` button zeroes the UI sliders (mjlab's Zero button, §3a).
    if (inputName !== 'zero') return;
    for (const input of this.cfg.ui?.inputs ?? []) {
      if (input.type === 'slider') this.uiValues.set(input.name, 0);
    }
  }

  /** Run one graph evaluation. Exposed for tests//deterministic stepping. */
  async step(resample: boolean): Promise<void> {
    const feeds: Record<string, OnnxTensorLike> = {};
    for (const spec of this.cfg.state_fields) {
      feeds[`prev_${spec.name}`] = this.state.get(spec.name)!;
    }
    for (const slot of this.cfg.input_slots ?? []) {
      const value = this.deps.readSlot?.(slot) ?? null;
      if (!value) continue;
      feeds[`${slot.entity ?? 'entity'}__${slot.field}`] = {
        data: value,
        dims: [1, value.length],
      };
    }
    feeds.resample_mask = { data: new Uint8Array([resample ? 1 : 0]), dims: [1] };
    feeds.rand = {
      data: this.deps.rng.randVector(this.cfg.rand_dim, this.cfg.rand_ranges),
      dims: [this.cfg.rand_dim],
    };

    const outputs = await this.deps.session.run(feeds);

    for (const spec of this.cfg.state_fields) {
      const next = outputs[`next_${spec.name}`];
      if (next) this.state.set(spec.name, next);
    }
    const commandTensor = this.state.get(this.cfg.command_field);
    if (commandTensor) this.command = toFloat32(commandTensor.data);

    this.applyWrites(outputs);
  }

  private applyWrites(outputs: Record<string, OnnxTensorLike>): void {
    const targets = this.cfg.write_targets ?? [];
    if (targets.length === 0) return;
    const mjModel = this.context?.mjModel;
    const mjData = this.context?.mjData;
    if (!mjModel || !mjData) return;
    const values: WriteValues = {};
    for (const [key, tensor] of Object.entries(outputs)) {
      if (!key.startsWith('next_')) values[key] = toFloat32(tensor.data);
    }
    applyEntityWrites(mjModel, mjData, targets, values);
  }

  private isUiEnabled(): boolean {
    const hasCheckbox = (this.cfg.ui?.inputs ?? []).some(
      i => i.type === 'checkbox' && i.name === 'enabled',
    );
    return hasCheckbox && (this.uiValues.get('enabled') ?? 0) > 0.5;
  }

  private sampleResampleTime(): number {
    const range = this.cfg.resampling_time_range;
    if (!range) return Number.POSITIVE_INFINITY; // resample only on reset
    return this.deps.rng.uniform(range[0], range[1]);
  }
}

function toFloat32(data: Float32Array | BigInt64Array | Uint8Array): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Number(data[i]);
  return out;
}
