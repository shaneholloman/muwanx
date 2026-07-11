/**
 * Public API surface for the headless mjswan engine (`createEngine`).
 *
 * This is the contract every layer builds against; see
 * docs/adr/0004-headless-engine-core.md §9. The engine runs one simulation at a
 * time, takes bytes directly (no fetching), and exposes switch verbs whose names
 * match the real cost (loadScene = rebuild; setPolicy/setSplat/setMotion = live).
 */
import type { ViewerConfig } from '../core/engine/viewer_config';
import type { ObservationConstructor } from '../core/policy/PolicyRunner';
import type { TerminationConstructor } from '../core/termination/terminations';
import type { EventConstructor } from '../core/event/EventBase';
import type { CommandTermConstructor } from '../core/command/types';

/** Asset bytes: already in hand, or a lazy loader fetched on demand. */
export type Bytes = ArrayBuffer | (() => Promise<ArrayBuffer>);

/**
 * Custom MDP terms registered into a pinned engine at load. Trusted contexts
 * only — mjswan Cloud rejects author code (ADR 0004 §10). Scene-scoped terms
 * (events) ride on {@link SceneInput}; policy-scoped terms (observations /
 * terminations / commands) on {@link PolicyInput}.
 */
export interface EnginePlugins {
  observations?: Record<string, ObservationConstructor>;
  terminations?: Record<string, TerminationConstructor>;
  events?: Record<string, EventConstructor>;
  commands?: Record<string, CommandTermConstructor>;
}

/** Spherical splat placement (a subset of the internal SplatConfig), in the splat's own frame. */
export interface SplatTransform {
  scale?: number;
  xOffset?: number;
  yOffset?: number;
  zOffset?: number;
  /** Degrees, applied on top of the COLMAP→Three.js base rotation. */
  roll?: number;
  pitch?: number;
  yaw?: number;
}

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
  motions?: MotionInput[];
  /** Policy-scoped custom terms (observations / terminations / commands). */
  plugins?: EnginePlugins;
}

export interface SceneInput {
  model: Bytes;           // .mjz (engine unpacks)
  policy?: PolicyInput | null;
  splat?: SplatInput | null;
  viewer?: ViewerConfig;
  /** Scene-scoped custom terms (events). */
  plugins?: EnginePlugins;
}

/** Camera pose in spherical MuJoCo coordinates (x forward, y left, z up). */
export interface CameraView {
  lookat: [number, number, number];
  distance: number;
  azimuth: number;
  elevation: number;
  fovy: number;
}

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
