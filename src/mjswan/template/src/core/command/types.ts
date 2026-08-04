import type * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';
import type { SeededRng } from '../rng';
import type { OnnxSessionCache, SlotReader } from '../onnx/session';

export type CommandType = 'slider' | 'button' | 'checkbox';

/**
 * A companion slider that rescales another slider's drag range (brief §3a).
 *
 * mjlab's play GUI pairs each velocity axis with a "Max <label>" slider whose only
 * job is to widen or narrow how far the value slider can be dragged. It is not
 * simulation state: nothing is sent to the engine, no command changes when it
 * moves. Purely how far the *other* control reaches.
 */
export interface SliderRangeControl {
  min: number;
  max: number;
  step: number;
  default: number;
  /** Label for the companion slider; defaults to `Max <label>`. */
  label?: string;
}

export interface SliderCommandConfig {
  type: 'slider';
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  enabled_when?: string;
  /**
   * When set, the app renders a companion range slider and clamps this one's
   * displayed range to `[-value, value]`. Symmetric around zero, matching the
   * three velocity axes mjlab does this for; an asymmetric range is a follow-up.
   */
  adjustable_range?: SliderRangeControl;
}

export interface ButtonCommandConfig {
  type: 'button';
  name: string;
  label: string;
}

export interface CheckboxCommandConfig {
  type: 'checkbox';
  name: string;
  label: string;
  default: boolean;
}

export type CommandInputConfig = SliderCommandConfig | ButtonCommandConfig | CheckboxCommandConfig;

export interface CommandUiConfig {
  inputs?: CommandInputConfig[];
}

export interface CommandConfigEntry {
  name: string;
  ui?: CommandUiConfig;
  [key: string]: unknown;
}

export type CommandsConfig = Record<string, CommandConfigEntry>;

export interface CommandDefinition {
  id: string;
  groupName: string;
  config: CommandInputConfig;
}

export function getCommandInputId(groupName: string, inputName: string): string {
  return `${groupName}:${inputName}`;
}

export type CommandEventType = 'change' | 'reset' | 'button' | 'group_registered' | 'clear';

export interface CommandEvent {
  type: CommandEventType;
  commandId: string;
  groupName?: string;
  value?: number;
}

export type CommandEventListener = (event: CommandEvent) => void;

export interface CommandTermContext {
  mujoco: MainModule;
  mjModel: MjModel | null;
  mjData: MjData | null;
  scene: THREE.Scene;
  bodies?: Record<number, THREE.Group> | null;
  mujocoRoot?: THREE.Group | null;
  requestReset?: () => void;
  /**
   * Deps for `OnnxCommand` terms (ADR 0005 §3): the orchestrator-owned seeded
   * PRNG and the loaded `.onnx` sessions for this policy's commands, keyed by
   * the same path `config.onnx` names. Absent for scenes with no ONNX commands.
   */
  rng?: SeededRng;
  onnxSessions?: OnnxSessionCache;
  /** Reads a command's declared dynamic runtime input slots (brief §3a). */
  readOnnxSlot?: SlotReader;
}

export interface CommandTerm {
  getCommand(): Float32Array;
  getUiConfig?(): CommandUiConfig | null;
  /**
   * Episode reset. May be async — mjlab's `CommandTerm.reset` *is* the resample
   * (`_resample(env_ids)`), and for a traced term that means an `ort.run()`.
   * `CommandManager.resetTerms` awaits it so the resample lands before the step's
   * single forward, as mjlab's does.
   */
  reset?(): void | Promise<void>;
  update?(dt: number): void;
  updateDebugVisuals?(): void;
  setValue?(inputName: string, value: number): number | void;
  triggerButton?(inputName: string): void;
  dispose?(): void;
}

export interface TrackingSource {
  isReady(): boolean;
  getAnchorBodyName(): string | null;
  getAnchorPos(frameIndex?: number): Float32Array | null;
  getAnchorQuat(frameIndex?: number): Float32Array | null;
  getBodyNames(): string[];
  getBodyPosW(frameIndex?: number): Float32Array | null;
}

export function isTrackingSource(term: unknown): term is TrackingSource {
  return (
    typeof term === 'object' &&
    term !== null &&
    'isReady' in term &&
    'getAnchorBodyName' in term &&
    'getBodyNames' in term
  );
}

export type CommandTermConstructor = new (
  termName: string,
  config: CommandConfigEntry,
  context: CommandTermContext
) => CommandTerm;
