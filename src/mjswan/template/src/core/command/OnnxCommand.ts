/**
 * The single generic command handler: every traced command is a data instantiation of
 * this class — a different graph, `state_fields`, `ui` and `write_targets`.
 *
 * The graph owns the math; this owns the native half — the scalar resample timer that
 * sets `resample_mask`, `prev_state` across frames, `rand` drawn from the seeded PRNG
 * (never ONNX's own random ops), the `entity_write` application, and the `viz` debug
 * marker any position-shaped state field gets for free.
 *
 * The UI override overwrites the command *after* the autonomous computation, which is
 * never skipped — mjlab's play-time behaviour.
 *
 * **Async boundary.** `update()`/`getCommand()` are sync but ORT-Web is not, so
 * `update()` kicks off inference and `getCommand()` serves the last completed value. A
 * frame arriving mid-flight is skipped, never queued, so no backlog can build.
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
  /** As the build found it. Absent means zero-fill, wrong for a counter or held value. */
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
    // Clamped to the declared width: a mismatch means config and graph disagree.
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
   * One traced state field, for a `{command, field}` slot on another term's graph. Raw
   * state, not `getCommand()`, since the UI override does not reach what mjlab reads.
   */
  getStateField(field: string): Float32Array | null {
    const tensor = this.state.get(field);
    // Copied: this tensor feeds the graph next frame, so in-place edits would corrupt it.
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
   * Resample **now**, as mjlab's `CommandTerm.reset` does — before the step's single
   * forward, so an `entity_write` it emits is published by that forward rather than
   * leaving the next observation on a stale `xpos`.
   *
   * The frame's later `update()` runs the same graph again with `resample_mask = 0`,
   * which is `_update_command` alone: mjlab's split across the forward, for one extra
   * `ort.run()` on reset frames.
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

  /** Move the marker to the current `viz.field` value; shown while `debug_vis`. */
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
    // mjlab's Zero button.
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
