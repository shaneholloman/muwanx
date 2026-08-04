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
 * - **Debug-vis marker** — when `viz` names a `state_fields` entry (a 3D
 *   position), a sphere is drawn there while `debug_vis` is true. Generic:
 *   any traced command with a position-shaped state field gets this for
 *   free, replacing what used to be a hand-written TS class per command
 *   (e.g. the retired `LiftingCommand.ts`).
 *
 * **Async boundary.** `CommandTerm.update()`/`getCommand()` are synchronous, but
 * ORT-Web inference is not. `update()` therefore *kicks off* inference and
 * returns; `getCommand()` serves the most recently completed value. A frame that
 * arrives while inference is still in flight is skipped rather than queued, so
 * the command can never build a backlog. The resulting one-frame lag is the same
 * property already accepted elsewhere in the design (ADR §8).
 */

import * as THREE from 'three';
import { SeededRng } from '../rng';
import { applyEntityWrites, type WriteTarget, type WriteValues } from '../event/entityWrite';
import { slotDims, slotInputName } from '../onnx/session';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import { mjcToThreeCoordinate } from '../scene/coordinate';
import type { CommandConfigEntry, CommandTerm, CommandTermContext, CommandUiConfig } from './types';

export type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader };

export interface OnnxStateFieldSpec {
  name: string;
  shape: number[];
  dtype: string;
  /**
   * Flattened initial value, as the build found it on the term (ADR 0005 §3).
   *
   * Optional on the read side only so an older bundle still loads: absent means
   * zero-fill, which is what this did before the build emitted the value. That is
   * correct for a term whose first resample overwrites every field — which is
   * every reference task today — and wrong for one holding a counter or a
   * carried-over value.
   */
  init?: number[];
}

export interface OnnxCommandVizConfig {
  /** Which `state_fields` entry to render — must be a 3D position. */
  field: string;
  shape: 'sphere';
  radius: number;
  /** RGBA, each in [0, 1]. */
  color: [number, number, number, number];
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
  viz?: OnnxCommandVizConfig;
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
  const init = spec.init;
  if (init) {
    // Truncated or padded to the declared width rather than trusted: a mismatch
    // means the config and the graph disagree, and writing past `data` would throw
    // while writing short would leave a silent zero in the middle of the state.
    for (let i = 0; i < Math.min(n, init.length); i++) data[i] = Number(init[i]);
  }
  return { data, dims: [...spec.shape] };
}

export class OnnxCommand implements CommandTerm {
  private readonly cfg: OnnxCommandConfig;
  private readonly deps: OnnxCommandDeps;
  private readonly context: CommandTermContext | null;

  private state = new Map<string, OnnxTensorLike>();
  private command: Float32Array;
  private timeLeft = 0;
  /** The running `step`, if any — `update` skips on it, `reset` waits for it. */
  private inFlight: Promise<void> | null = null;
  /** Set on the first update so the initial frame resamples (reset semantics). */
  private pendingResample = true;
  private uiValues = new Map<string, number>();
  private readonly marker: THREE.Mesh | null;

  constructor(
    termName: string,
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
    this.marker = config.viz && context ? this.createMarker(termName, config.viz) : null;
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

  /**
   * One of this command's traced state fields, for a `{command, field}` input
   * slot on another term's graph (mjlab's `object_to_goal_distance` measures
   * against the lift command's `target_pos`). Null for an undeclared field.
   *
   * Deliberately the raw state, not `getCommand()`: the UI override applies to
   * the command vector a policy consumes, while a slot has to mirror what mjlab
   * reads off the command term itself.
   */
  getStateField(field: string): Float32Array | null {
    const tensor = this.state.get(field);
    // Copied: `toFloat32` passes a Float32Array straight through, and this is the
    // tensor the graph is fed next frame — a caller scaling it in place would
    // corrupt the command's own state.
    return tensor ? Float32Array.from(toFloat32(tensor.data)) : null;
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
    void this.run(resample);
  }

  /**
   * Resample **now**, as mjlab's `CommandTerm.reset` does.
   *
   * mjlab calls `_resample(env_ids)` from inside `_reset_idx`, which runs *before*
   * the step's single `sim.forward()`; `_update_command` then runs after it, from
   * `command_manager.compute(dt)`. This used to only raise a flag, so the whole graph
   * ran on the far side of the forward instead. Two consequences, both divergences
   * from the env the policy was trained in: the slots the resample reads saw fresher
   * derived state than mjlab's do, and — the one that bites — an `entity_write` it
   * emits (a lifted object's pose) landed *after* the forward meant to publish it,
   * leaving the next observation reading a stale `xpos` for that body.
   *
   * Awaited by `CommandManager.resetTerms`, which the reset chain awaits before the
   * forward. The frame's later `update()` then runs the graph again with
   * `resample_mask = 0`, which is `_update_command` alone — exactly mjlab's split
   * across the forward, for one extra `ort.run()` on reset frames only.
   *
   * ADR §3's "reset unifies to `resample_mask = true`" still holds: there is no
   * separate reset path in the graph, only a second call to the same one.
   */
  async reset(): Promise<void> {
    this.timeLeft = this.sampleResampleTime();
    this.pendingResample = false;
    // Never interleaved with a step already running: both read and rewrite `state`.
    await this.inFlight?.catch(() => {});
    await this.run(true);
  }

  /** One `step`, tracked so `update` can skip it and `reset` can wait for it. */
  private run(resample: boolean): Promise<void> {
    const pending = this.step(resample).finally(() => {
      if (this.inFlight === pending) this.inFlight = null;
    });
    this.inFlight = pending;
    return pending;
  }

  /** Move the marker to the current `viz.field` state value; visible only
   * while `debug_vis` is set (generic — see class docs). */
  updateDebugVisuals(): void {
    if (!this.marker || !this.cfg.viz) return;
    this.marker.visible = Boolean(this.cfg.debug_vis);
    if (!this.marker.visible) return;
    const tensor = this.state.get(this.cfg.viz.field);
    if (!tensor) return;
    this.marker.position.copy(mjcToThreeCoordinate(toFloat32(tensor.data)));
  }

  dispose(): void {
    if (!this.marker) return;
    this.context?.scene.remove(this.marker);
    this.marker.geometry.dispose();
    const material = this.marker.material;
    if (Array.isArray(material)) {
      for (const entry of material) entry.dispose();
    } else {
      material.dispose();
    }
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
      feeds[slotInputName(slot)] = { data: value, dims: slotDims(slot, value.length) };
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

  private createMarker(termName: string, viz: OnnxCommandVizConfig): THREE.Mesh {
    const geometry = new THREE.SphereGeometry(viz.radius, 20, 12);
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(viz.color[0], viz.color[1], viz.color[2]),
      transparent: true,
      opacity: viz.color[3],
      depthWrite: false,
    });
    const marker = new THREE.Mesh(geometry, material);
    marker.name = `mjswan-command-${termName}-viz`;
    marker.visible = false;
    this.context!.scene.add(marker);
    return marker;
  }
}

function toFloat32(data: Float32Array | BigInt64Array | Uint8Array): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Number(data[i]);
  return out;
}
