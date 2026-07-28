/**
 * Public API surface for the headless mjswan engine (`createEngine`).
 *
 * This is the contract every layer builds against; see
 * docs/adr/0004-headless-engine-core.md §9. The engine runs one simulation at a
 * time, takes bytes directly (no fetching), and exposes switch verbs whose names
 * match the real cost (loadScene = rebuild; setPolicy/setSplat/setMotion = live).
 */
import type { CameraView, ViewerConfig } from '../core/engine/viewer_config';
import type { EventConfig, TerrainData } from '../core/event/EventBase';
import type { Bytes } from '../core/utils/bytes';
import type { SplatTransform } from '../core/scene/splat';
import type { EnginePlugins } from '../core/plugins';

// Bytes: asset bytes in hand or a lazy loader. SplatTransform: spherical splat placement.
// EnginePlugins: custom MDP term constructors (trusted-only; ADR 0004 §10).
export type { Bytes, SplatTransform, EnginePlugins };

export interface SplatInput {
  data: Bytes;            // .spz
  collider?: Bytes;       // optional collider mesh
  transform?: SplatTransform;
}

export interface MotionInput {
  name: string;
  data: Bytes;            // .npz (lazy loader preserves today's on-demand load)
  default?: boolean;
}

export interface PolicyInput {
  /** Parsed policy.json; opaque to the app, interpreted by the engine. */
  config: object;
  onnx: Bytes;
  /**
   * Traced term-body graphs, keyed by the path the config refers to them by
   * (`"obs/joint_pos.onnx"`, `"term/fell_over.onnx"`, `"command/twist.onnx"`).
   *
   * ADR 0005 gives every traced observation / termination / command term its own
   * small graph alongside the policy network, and the engine never fetches, so
   * the app delivers their bytes here. `mjswan/manifest` fills this in from
   * `policy.json`; a hand-assembled `PolicyInput` can use
   * `policyGraphRefs(config)` to enumerate what to load. A missing entry does not
   * fail the load — the manager that wanted it warns and skips that one term.
   */
  graphs?: Record<string, Bytes>;
  motions?: MotionInput[];
  /** Policy-scoped custom terms (observations / terminations / commands). */
  plugins?: EnginePlugins;
}

export interface SceneInput {
  model: Bytes;           // .mjz (engine unpacks)
  policy?: PolicyInput | null;
  splat?: SplatInput | null;
  viewer?: ViewerConfig;
  /** Declarative reset events (e.g. terrain randomization) + their terrain data. */
  events?: EventConfig[];
  terrainData?: TerrainData;
  /** Traced event-term graphs (`"event/push_robot.onnx"`), as {@link PolicyInput.graphs}. */
  graphs?: Record<string, Bytes>;
  /** Scene-scoped custom terms (events). */
  plugins?: EnginePlugins;
}

/** Camera pose in spherical MuJoCo coordinates (x forward, y left, z up). */
export type { CameraView };

export interface CameraControls {
  /** Overwrite the current pose; body tracking (if any) continues, user drag stays live. */
  set(view: Partial<CameraView>): void;
  get(): CameraView;
  /** Re-fit the camera to the scene bounds. */
  frame(): void;
}

/** A policy command term surfaced for generic UI controls. */
export interface CommandDescriptor {
  id: string;             // "group:name"
  group: string;
  type: 'slider' | 'checkbox' | 'button';
  label: string;
  min?: number;           // slider only
  max?: number;           // slider only
  step?: number;          // slider only
  /** Slider only: name of a sibling checkbox that gates this control. */
  enabledWhen?: string;
  /**
   * Slider only: a companion control that rescales this slider's drag range
   * (brief §3a). Entirely presentational — the app clamps the displayed range to
   * `[-value, value]` locally and never calls `set` for it, so moving it changes
   * no simulation state. Absent unless the build asked for one.
   */
  adjustableRange?: SliderRangeControl;
}

/** Bounds of an {@link CommandDescriptor.adjustableRange} companion slider. */
export interface SliderRangeControl {
  min: number;
  max: number;
  step: number;
  default: number;
  label?: string;
}

export interface CommandControls {
  set(id: string, value: number): void;
  trigger(id: string): void;
}

/** Immutable snapshot pushed to {@link MjswanEngine.subscribe} listeners. */
export interface MjswanEngineState {
  phase: 'running' | 'paused';
  loading: boolean;
  loadingMessage: string | null;
  error: Error | null;
  commands: ReadonlyArray<CommandDescriptor>;
  commandValues: Readonly<Record<string, number>>;
}

export interface CreateEngineOptions {
  /** Load `mujoco/mt` (SharedArrayBuffer; requires COOP/COEP). Default false. */
  multithreaded?: boolean;
}

/** A headless, instance-scoped simulation engine. Create with {@link createEngine}. */
export interface MjswanEngine {
  // content — verbs match switch cost
  loadScene(input: SceneInput): Promise<void>;          // full model rebuild
  setPolicy(input: PolicyInput | null): Promise<void>;  // live, keeps model
  setSplat(input: SplatInput | null): Promise<void>;    // live
  setMotion(name: string | null): Promise<boolean>;     // live; returns whether accepted
  setReferenceVisible(visible: boolean): void;          // motion ghost toggle
  /** Live-update the current splat's placement (dev calibration; no reload). */
  calibrateSplat(transform: SplatTransform): void;

  // playback
  play(): void;
  pause(): void;
  reset(): void;

  // subsystems
  readonly camera: CameraControls;
  readonly commands: CommandControls;

  // state
  getState(): MjswanEngineState;
  subscribe(listener: (state: MjswanEngineState) => void): () => void;

  // misc
  captureThumbnail(options?: { maxDim?: number; quality?: number }): Promise<Blob>;
  dispose(): void;
}
