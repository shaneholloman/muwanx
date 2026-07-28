import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import type { MainModule, MjData, MjModel } from 'mujoco';
import {
  getPosition,
  getQuaternion,
  loadSceneFromURL,
} from '../scene/scene';
import { loadMjzFile } from '../utils/mjzLoader';
import { type Bytes } from '../utils/bytes';
import type { EnginePlugins } from '../plugins';
import { type SplatTransform, type SplatMesh, loadSplat, disposeSplat, applySplatTransform } from '../scene/splat';
import { loadCollider, disposeCollider } from '../scene/collider';
import { DragStateManager } from '../utils/dragStateManager';
import { createTendonState, updateTendonGeometry, updateTendonRendering } from '../scene/tendons';
import { updateHeadlightFromCamera, updateLightsFromData } from '../scene/lights';
import { mjcToThreeCoordinate, threeToMjcCoordinate } from '../scene/coordinate';
import {
  type CameraView,
  type ViewerConfig,
  type ViewerState,
  applyViewerConfig,
  computeCameraPosition,
  updateCameraFromData,
} from './viewer_config';
import { Observations } from '../observation/observations';
import { TerminationManager } from '../termination/TerminationManager';
import { Terminations } from '../termination/terminations';
import * as ort from 'onnxruntime-web';
import { PolicyRunner } from '../policy/PolicyRunner';
import { OnnxModule } from '../policy/OnnxModule';
import { PolicyStateBuilder } from '../policy/PolicyStateBuilder';
import type { PolicyConfig } from '../policy/types';
import { TrackingPolicy } from '../policy/modules/TrackingPolicy';
import { LocomotionPolicy } from '../policy/modules/LocomotionPolicy';
import { CommandManager, type CommandTermContext, type CommandsConfig } from '../command';
import { EventManager } from '../event/EventManager';
import { Events } from '../event/events';
import type { EventConfig, EventContext, TerrainData } from '../event/EventBase';
import { OnnxSessionCache, type SlotReader } from '../onnx/session';
import type { RaycastSensorDescriptor } from '../onnx/raycast';
import { createSlotReader } from '../onnx/slotReader';
import { SeededRng } from '../rng';

/**
 * Seed for the term PRNG (ADR 0005 §2). Fixed rather than time-derived so a plain
 * page load is reproducible; a "share this run" feature overrides it per session.
 */
const DEFAULT_TERM_SEED = 0x5eed;

/**
 * Encoder bias by joint name, for the slot reader's `joint_pos_biased`.
 *
 * mjlab randomizes `encoder_bias` per episode and `joint_pos_biased` is what a
 * trained policy observed; the export captures the bias into `policy.json`
 * aligned to `policy_joint_names`. Keyed by name because the reader needs it in
 * *entity*-joint order, which is not the action order this array is in. Empty
 * when the config carries no bias — mjlab's own zero-bias default, under which
 * `joint_pos_biased` is just `joint_pos`.
 */
function buildJointBias(config: PolicyConfig): Map<string, number> {
  const bias = new Map<string, number>();
  const values = config.encoder_bias;
  const names = config.policy_joint_names;
  if (!Array.isArray(values) || !Array.isArray(names)) return bias;
  for (let i = 0; i < names.length && i < values.length; i++) {
    if (values[i]) bias.set(names[i], values[i]);
  }
  return bias;
}

/**
 * Structured-sensor descriptors from every observation group in a policy config.
 *
 * They ride on the group rather than the policy because only a fused group knows
 * which sensors its slots name; gathering them here keeps the slot reader from
 * having to know about groups at all.
 */
function collectRaycastSensors(
  config: PolicyConfig
): Record<string, RaycastSensorDescriptor> {
  const sensors: Record<string, RaycastSensorDescriptor> = {};
  for (const group of Object.values(config.observations ?? {})) {
    const declared = (group as { sensors?: Record<string, RaycastSensorDescriptor> })
      .sensors;
    if (declared) Object.assign(sensors, declared);
  }
  return sensors;
}

/** A policy with its ONNX weights resolved to bytes; motion data stays lazy. */
export type ResolvedPolicy = {
  config: PolicyConfig;
  onnx: ArrayBuffer;
  /** Traced term graphs, keyed by the config-relative path (ADR 0005 §4). */
  graphs?: Array<{ name: string; data: ArrayBuffer }>;
  motions: Array<{ name: string; data: Bytes; default?: boolean }>;
  /** Policy-scoped custom terms (observations / terminations / commands). */
  plugins?: EnginePlugins;
};

/** A splat with its bytes resolved. */
export type ResolvedSplat = {
  data: ArrayBuffer;
  collider?: ArrayBuffer | null;
  transform?: SplatTransform;
};

/** A full scene ready to build: model bytes plus resolved policy/splat/config. */
export type ResolvedScene = {
  model: ArrayBuffer;
  policy?: ResolvedPolicy | null;
  splat?: ResolvedSplat | null;
  viewer?: ViewerConfig | null;
  events?: EventConfig[] | null;
  terrainData?: TerrainData | null;
  /** Traced event-term graphs, keyed by the model-relative path (ADR 0005 §4). */
  graphs?: Array<{ name: string; data: ArrayBuffer }>;
  /** Scene-scoped custom terms (events). */
  plugins?: EnginePlugins;
};

type MotionCommandTerm = {
  setSelectedMotion(name: string | null): Promise<boolean> | boolean;
  setReferenceVisible?(visible: boolean): void;
  getSelectedMotionName?(): string | null;
};

function isMotionCommandTerm(term: unknown): term is MotionCommandTerm {
  return (
    typeof term === 'object' &&
    term !== null &&
    typeof (term as MotionCommandTerm).setSelectedMotion === 'function'
  );
}

/** Thrown when a scene exceeds the browser's 2 GB WebAssembly memory limit. */
export class WasmMemoryLimitError extends Error {
  constructor() {
    super(
      "This scene cannot be loaded because it exceeds the browser's " +
        'WebAssembly 2 GB memory limit. ' +
        'Try closing other browser tabs or reloading the page to free memory.'
    );
    this.name = 'WasmMemoryLimitError';
  }
}

// mj_loadXML surfaces OOM via four distinct paths depending on which internal
// allocator hits the 2 GB ceiling first (null return, MuJoCo error string,
// lodepng error string, or raw bad_alloc).
function isWasmOom(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    msg.includes('MjModel loading returned null') ||
    msg.includes('Could not allocate memory') ||
    msg.includes('memory allocation failed') ||
    msg.includes('bad_alloc')
  );
}

type BodyState = {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
};

export class mjswanRuntime {
  private mujoco: MainModule;
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private mjModel: MjModel | null;
  private mjData: MjData | null;
  private bodies: Record<number, THREE.Group> | null;
  private lights: THREE.Light[];
  private mujocoRoot: THREE.Group | null;
  private lastSimState: {
    bodies: Map<number, BodyState>;
    tendons: ReturnType<typeof createTendonState>;
  };
  private dynamicBodyIds: Set<number> | null;
  private loopPromise: Promise<void> | null;
  private running: boolean;
  private timestep: number;
  private decimation: number;
  private policyCtrlDt: number | null;
  private loadingScene: Promise<void> | null;
  private resizeObserver: ResizeObserver | null;
  private dragStateManager: DragStateManager | null;
  private dragForceScale: number;
  private policyRunner: PolicyRunner | null;
  private policyStateBuilder: PolicyStateBuilder | null;
  private initialQpos: number[] | null;
  private initialQvel: number[] | null;
  private policyControl: Array<{
    controlType: string;
    ctrlAdr: number[];
    qposAdr: number[];
    qvelAdr: number[];
    // Indices into the flat policy action vector for this term's joints.
    actionIndices: number[];
    actionScale: Float32Array;
    actionOffset: Float32Array;
    defaultJointPos: Float32Array;
    encoderBias: Float32Array;
    // Per-actuator flag: true = position actuator (ctrl=target_pos, PD internal),
    // false = motor actuator (ctrl=torque, PD computed in browser from kp/kd).
    positionActuator: boolean[];
    kp: Float32Array;
    kd: Float32Array;
    // muscle_activation only: when true apply MyoSuite sigmoid; when false clip(raw, 0, 1).
    muscleNormalize: boolean;
  }> | null;
  private onnxModule: OnnxModule | null;
  private onnxInputDict: Record<string, ort.Tensor> | null;
  private onnxInferencing: boolean;
  private onnxTimeStep: number;
  private terminationManager: TerminationManager | null;
  private eventManager: EventManager | null;
  private terrainData: TerrainData | null;
  private vrButton: HTMLElement | null;
  private splatMesh: SplatMesh | null;
  private colliderMesh: THREE.Group | null;
  private currentSplatTransform: SplatTransform;
  private cameraState: ViewerState;
  private commandManager: CommandManager;
  // Instance-scoped custom terms: scene-scoped (events) and policy-scoped
  // (observations / terminations / commands). No module-global registries.
  private scenePlugins: EnginePlugins;
  private policyPlugins: EnginePlugins;
  /**
   * Traced term-body graphs (ADR 0005 §4), split by lifetime: event graphs belong
   * to the scene, everything else to the policy, so `setPolicy` replaces one
   * without disturbing the other.
   */
  private policyGraphs: OnnxSessionCache;
  private sceneGraphs: OnnxSessionCache;
  /**
   * The single seeded PRNG every ONNX term draws `rand` from (ADR 0005 §2), so a
   * session replays bit-for-bit. Scene-lifetime: reseeded per `loadEnvironment`.
   */
  private termRng: SeededRng;
  /** Reads the dynamic state a traced graph declares as inputs (ADR 0005 §6). */
  private readonly readOnnxSlot: SlotReader;
  /** Encoder bias by joint name, from the active policy config; see `SlotReaderOptions`. */
  private jointBias = new Map<string, number>();
  /**
   * Descriptors for structured sensors the policy's slots name (mjlab's height
   * scan). Policy-scoped, since they come out of `policy.json`'s groups.
   */
  private raycastSensors: Record<string, RaycastSensorDescriptor> = {};

  constructor(mujoco: MainModule, container: HTMLElement) {
    this.mujoco = mujoco;
    this.container = container;
    this.commandManager = new CommandManager();
    this.scenePlugins = {};
    this.policyPlugins = {};
    this.policyGraphs = new OnnxSessionCache();
    this.sceneGraphs = new OnnxSessionCache();
    this.termRng = new SeededRng(DEFAULT_TERM_SEED);
    // Reads `this.mjModel`/`this.mjData` per call rather than capturing them, so a
    // scene rebuild is picked up without rewiring every manager.
    this.readOnnxSlot = createSlotReader(
      () => ({
        mujoco: this.mujoco,
        mjModel: this.mjModel,
        mjData: this.mjData,
        commandManager: this.commandManager,
      }),
      {
        jointBias: (name) => this.jointBias.get(name) ?? 0,
        raycastSensors: () => this.raycastSensors,
      },
    );

    const workingPath = '/working';
    try {
      this.mujoco.FS.mkdir(workingPath);
    } catch (error: unknown) {
      if (error && typeof error === 'object' && 'code' in error && error.code !== 'EEXIST') {
        console.warn('Failed to create /working directory:', error);
      }
    }
    try {
      this.mujoco.FS.mount(this.mujoco.MEMFS, { root: '.' }, workingPath);
    } catch (error: unknown) {
      if (error && typeof error === 'object' && 'code' in error && error.code !== 'EEXIST' && error.code !== 'EBUSY') {
        console.warn('Failed to mount MEMFS at /working:', error);
      }
    }

    const { width, height } = this.getSize();

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 1000);
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.xr.enabled = true;
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(width, height);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.container.appendChild(this.renderer.domElement);

    this.vrButton = null;
    navigator.xr?.isSessionSupported('immersive-vr').then((supported) => {
      if (supported) {
        this.vrButton = VRButton.createButton(this.renderer);
        document.body.appendChild(this.vrButton);
      }
    });

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.2, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    this.renderer.setAnimationLoop(this.render);
    window.addEventListener('resize', this.onWindowResize);

    if ('ResizeObserver' in window) {
      this.resizeObserver = new ResizeObserver(() => this.onWindowResize());
      this.resizeObserver.observe(this.container);
    } else {
      this.resizeObserver = null;
    }

    this.lastSimState = {
      bodies: new Map(),
      tendons: createTendonState(),
    };
    this.dynamicBodyIds = null;

    this.mjModel = null;
    this.mjData = null;
    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.loopPromise = null;
    this.running = false;
    this.timestep = 0.001;
    this.decimation = 1;
    this.policyCtrlDt = null;
    this.loadingScene = null;
    this.dragStateManager = null;
    this.dragForceScale = 100.0;
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.initialQpos = null;
    this.initialQvel = null;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.onnxTimeStep = 0;
    this.terminationManager = null;
    this.eventManager = null;
    this.terrainData = null;
    this.splatMesh = null;
    this.colliderMesh = null;
    this.currentSplatTransform = {};
    this.cameraState = { trackBodyId: null, prevBodyPos: null };
  }

  async loadEnvironment(scene: ResolvedScene): Promise<void> {
    this.scenePlugins = scene.plugins ?? {};
    this.terrainData = scene.terrainData ?? null;
    // A fresh scene is a fresh run: reseed so two loads of the same scene draw the
    // same randomness (ADR 0005 §2).
    this.termRng = new SeededRng(DEFAULT_TERM_SEED);
    this.sceneGraphs.clear();
    await this.sceneGraphs.load(scene.graphs ?? []);
    if (scene.events && scene.events.length > 0) {
      this.eventManager = new EventManager(
        scene.events,
        { ...Events, ...this.scenePlugins.events },
        {
          sessions: this.sceneGraphs,
          rng: this.termRng,
          readSlot: this.readOnnxSlot,
        },
      );
      console.log(
        `[EventManager] ${this.eventManager.size} event term(s) loaded ` +
          `(${this.sceneGraphs.size} traced graph(s))`,
      );
    } else {
      this.eventManager = null;
    }
    await this.stop();

    // Dispose previous splat/collider before switching scenes
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }

    // Initialize CommandManager with default velocity commands
    this.initializeCommands();

    // Clear current references before loading the new scene.
    this.mjModel = null;
    this.mjData = null;
    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.dynamicBodyIds = null;

    await this.buildSceneFromMjz(scene.model);

    if (scene.splat) {
      await this.applySplat(scene.splat);
    }

    await this.loadPolicyConfig(scene.policy ?? null);

    this.applyViewerConfig(scene.viewer ?? null);

    // `mode="startup"` terms fire once, after the model and policy exist — they
    // write to mjData (domain randomization), so there has to be an mjData.
    if (this.eventManager && this.mjModel && this.mjData) {
      await this.eventManager.startup(this.eventContext());
      this.mujoco.mj_forward(this.mjModel, this.mjData);
      this.updateCachedState();
    }

    this.running = true;
    void this.startLoop();
  }

  /** Context every event term writes through; re-read per call (scene reloads). */
  private eventContext(): EventContext {
    return {
      mjModel: this.mjModel,
      mjData: this.mjData,
      terrainData: this.terrainData,
    };
  }

  /**
   * Initialize the CommandManager (clear).
   * Commands are registered from policy config in loadPolicyConfig().
   */
  private initializeCommands(): void {
    this.commandManager.clear();
  }

  /**
   * Initialize commands from policy config
   */
  private initializeCommandsFromConfig(
    commands: CommandsConfig,
    context: CommandTermContext
  ): void {
    const commandManager = this.commandManager;
    commandManager.initialize(commands, context, this.policyPlugins.commands);
    console.log('[mjswanRuntime] Commands loaded from policy config:', Object.keys(commands));
  }

  /**
   * Public method to reset the simulation state
   * Can be called from UI components via the CommandManager
   */
  resetSimulation(): void {
    this.resetSimulationState();
    if (this.policyRunner && this.policyStateBuilder) {
      const state = this.policyStateBuilder.build();
      this.policyRunner.reset(state);
    }
    console.log('[mjswanRuntime] Simulation reset');
  }

  async setSelectedMotion(motionName: string | null): Promise<boolean> {
    const term = this.commandManager.getTerm('motion');
    if (!isMotionCommandTerm(term)) {
      return false;
    }
    const accepted = await term.setSelectedMotion(motionName);
    if (accepted) {
      this.resetSimulation();
    }
    return accepted;
  }

  getSelectedMotionName(): string | null {
    const term = this.commandManager.getTerm('motion');
    if (!isMotionCommandTerm(term)) {
      return null;
    }
    return term.getSelectedMotionName?.() ?? null;
  }

  setReferenceVisible(visible: boolean): void {
    const term = this.commandManager.getTerm('motion');
    if (!isMotionCommandTerm(term) || typeof term.setReferenceVisible !== 'function') {
      return;
    }
    term.setReferenceVisible(visible);
  }

  private async buildScene(xmlPath: string): Promise<void> {
    if (this.loadingScene) {
      await this.loadingScene;
    }

    this.loadingScene = (async () => {
      const existingRoot = this.scene.getObjectByName('MuJoCo Root');
      if (existingRoot) {
        this.scene.remove(existingRoot);
      }

      const parent = {
        mjModel: this.mjModel,
        mjData: this.mjData,
        scene: this.scene,
      };

      [this.mjModel, this.mjData, this.bodies, this.lights] = await loadSceneFromURL(
        this.mujoco,
        xmlPath,
        parent
      );

      if (!this.mjModel || !this.mjData) {
        throw new Error('Failed to load MuJoCo model.');
      }

      this.mujocoRoot = this.scene.getObjectByName('MuJoCo Root') as THREE.Group | null;

      this.mujoco.mj_forward(this.mjModel, this.mjData);
      updateLightsFromData(this.mujoco, this.mjData, this.lights);
      updateHeadlightFromCamera(this.camera, this.lights);
      this.dynamicBodyIds = this.computeDynamicBodyIds(this.mjModel);
      this.syncStaticBodiesFromData();

      this.timestep = this.mjModel.opt.timestep || 0.001;
      const ctrlDtForDec = this.policyCtrlDt ?? 0.02;
      this.decimation = Math.max(1, Math.round(ctrlDtForDec / this.timestep));

      this.lastSimState.bodies.clear();
      this.updateCachedState();

      // Initialize DragStateManager
      if (!this.dragStateManager) {
        this.dragStateManager = new DragStateManager({
          scene: this.scene,
          renderer: this.renderer,
          camera: this.camera,
          container: this.container,
          controls: this.controls,
          draggableBodyIds: this.dynamicBodyIds,
        });
      } else {
        this.dragStateManager.setDraggableBodyIds(this.dynamicBodyIds);
      }

      this.loadingScene = null;
    })();

    await this.loadingScene;
  }

  // Unpack the app-supplied .mjz bytes into the VFS, then build the scene.
  // No scene cache to reclaim on OOM, so surface it directly as WasmMemoryLimitError.
  private async buildSceneFromMjz(model: ArrayBuffer): Promise<void> {
    try {
      const xmlPath = await loadMjzFile(this.mujoco, model);
      await this.buildScene(xmlPath);
    } catch (error) {
      this.loadingScene = null;
      if (isWasmOom(error)) {
        throw new WasmMemoryLimitError();
      }
      throw error;
    }
  }

  async startLoop(): Promise<void> {
    if (this.loopPromise) {
      return this.loopPromise;
    }
    this.running = true;
    this.loopPromise = this.mainLoop();
    return this.loopPromise;
  }

  async setSplat(splat: ResolvedSplat | null): Promise<void> {
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }
    if (splat) {
      await this.applySplat(splat);
    }
  }

  private async applySplat(splat: ResolvedSplat): Promise<void> {
    this.currentSplatTransform = splat.transform ?? {};
    this.splatMesh = loadSplat(splat.data, this.currentSplatTransform, this.scene);
    if (splat.collider) {
      this.colliderMesh = await loadCollider(splat.collider, this.scene);
    }
  }

  /** Update transform of the existing splat without disposing/reloading (dev calibration). */
  calibrateSplat(transform: SplatTransform): void {
    this.currentSplatTransform = transform;
    if (this.splatMesh) {
      applySplatTransform(this.splatMesh, transform);
    }
  }

  setSplatVisible(visible: boolean): void {
    if (this.splatMesh) {
      this.splatMesh.visible = visible;
    }
  }

  // ── playback + commands (driven by the engine's verbs) ──────────────────
  /** Instance-scoped command manager the engine reads/writes and subscribes to. */
  get commands(): CommandManager {
    return this.commandManager;
  }

  /** Resume the physics loop (rendering runs continuously regardless). */
  play(): void {
    this.running = true;
    void this.startLoop();
  }

  /** Halt physics; rendering continues so a frozen frame can be orbited. */
  pause(): void {
    void this.stop();
  }

  get isRunning(): boolean {
    return this.running;
  }

  // ── camera (spherical, MuJoCo coordinates) ──────────────────────────────
  /** Read the current camera pose back in spherical MuJoCo coordinates. */
  getCameraView(): CameraView {
    const offsetThree = this.camera.position.clone().sub(this.controls.target);
    const distance = offsetThree.length();
    const offset = threeToMjcCoordinate(offsetThree);
    const target = threeToMjcCoordinate(this.controls.target.clone());
    const RAD2DEG = 180 / Math.PI;
    const elevation = distance > 1e-9 ? Math.asin(-offset.z / distance) * RAD2DEG : 0;
    const azimuth = Math.atan2(offset.y, offset.x) * RAD2DEG;
    return {
      lookat: [target.x, target.y, target.z],
      distance,
      azimuth,
      elevation,
      fovy: this.camera.fov,
    };
  }

  /**
   * Overwrite the camera pose. Body tracking (if any) keeps following after
   * this — it applies a per-frame delta — and OrbitControls stay live.
   */
  setCameraView(view: Partial<CameraView>): void {
    const cur = this.getCameraView();
    const lookat = view.lookat ?? cur.lookat;
    const distance = view.distance ?? cur.distance;
    const elevation = view.elevation ?? cur.elevation;
    const azimuth = view.azimuth ?? cur.azimuth;
    this.camera.fov = view.fovy ?? cur.fovy;
    this.camera.updateProjectionMatrix();
    this.camera.position.copy(computeCameraPosition(lookat, distance, elevation, azimuth));
    this.controls.target.copy(mjcToThreeCoordinate(lookat));
    this.controls.update();
  }

  /** Re-fit the camera to the current scene bounds, keeping azimuth/elevation. */
  frameCamera(): void {
    if (!this.mujocoRoot) {
      return;
    }
    const box = new THREE.Box3().setFromObject(this.mujocoRoot);
    if (box.isEmpty()) {
      return;
    }
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    const fov = (this.camera.fov * Math.PI) / 180;
    const fitDistance = size / (2 * Math.tan(fov / 2));
    const cur = this.getCameraView();
    const lookat = threeToMjcCoordinate(center);
    this.setCameraView({
      lookat: [lookat.x, lookat.y, lookat.z],
      distance: fitDistance > 0 ? fitDistance : cur.distance,
    });
  }

  async stop(): Promise<void> {
    this.running = false;
    const pending = this.loopPromise;
    if (pending) {
      await pending;
    }
    this.loopPromise = null;
  }

  private async mainLoop(): Promise<void> {
    while (this.running) {
      const loopStart = performance.now();
      const target = this.timestep * this.decimation;

      if (this.mjModel && this.mjData) {
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        if (this.policyRunner && this.policyStateBuilder) {
          const state = this.policyStateBuilder.build();
          const obs = await this.policyRunner.collectObservationsByKey(state);
          await this.runOnnxInference(obs);
        }
        this.executeSimulationSteps();
        this.updateCachedState();

        // Evaluate termination conditions after simulation step
        if (this.terminationManager && this.policyStateBuilder) {
          const postState = this.policyStateBuilder.build();
          const result = this.terminationManager.evaluate(postState, target);
          if (result.done) {
            this.resetSimulationState();
            this.terminationManager.reset();
            if (this.policyRunner) {
              const resetState = this.policyStateBuilder.build();
              this.policyRunner.reset(resetState);
            }
          }
        }

        // Commands are updated after the physics step to match mjlab training-time semantics:
        // the policy sees command values from the previous step, consistent with how mjlab
        // computes observations before stepping the environment.
        this.commandManager.update(target);
        this.commandManager.updateDebugVisuals();
        // Advances `mode="interval"` timers (mjlab's mid-episode randomization) and
        // the reset gates' step counters.
        this.eventManager?.tick(target, this.eventContext());
      }

      const elapsed = (performance.now() - loopStart) / 1000;
      const sleepTime = Math.max(0, target - elapsed);
      if (sleepTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, sleepTime * 1000));
      }
    }
    this.loopPromise = null;
  }

  async loadPolicyConfig(policy: ResolvedPolicy | null): Promise<void> {
    this.policyPlugins = policy?.plugins ?? {};
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.onnxTimeStep = 0;
    this.terminationManager = null;
    this.policyGraphs.clear();
    this.jointBias.clear();
    this.raycastSensors = {};
    // Note: eventManager, sceneGraphs and terrainData are scene-level state set in
    // loadEnvironment; do not clear here.

    // Clear existing commands when switching policies
    this.commandManager.clear();

    if (!policy) {
      return;
    }

    if (!this.mjModel || !this.mjData) {
      console.warn('Policy config loaded before MuJoCo model is ready.');
      return;
    }

    try {
      const config = policy.config;
      await this.policyGraphs.load(policy.graphs ?? []);
      this.jointBias = buildJointBias(config);
      this.raycastSensors = collectRaycastSensors(config);
      // Motion metadata lives in policy.json; the app supplies the bytes as
      // MotionInput[]. Merge the bytes onto each motion entry by name.
      if (Array.isArray(config.motions)) {
        const dataByName = new Map(policy.motions.map((m) => [m.name, m.data]));
        config.motions = config.motions.map((motion) => ({
          ...motion,
          data: dataByName.get(motion.name),
        }));
      }
      if (config.commands?.motion && Array.isArray(config.motions)) {
        config.commands.motion = {
          ...config.commands.motion,
          motions: config.motions,
        };
      }
      this.initialQpos = Array.isArray(config.initial_qpos) ? (config.initial_qpos as number[]) : null;
      this.initialQvel = Array.isArray(config.initial_qvel) ? (config.initial_qvel as number[]) : null;
      this.resetSimulationState();
      this.mujoco.mj_forward(this.mjModel, this.mjData);
      this.updateCachedState();

      // Initialize commands from policy config if present
      if (config.commands && typeof config.commands === 'object') {
        this.initializeCommandsFromConfig(config.commands as CommandsConfig, {
          mujoco: this.mujoco,
          mjModel: this.mjModel,
          mjData: this.mjData,
          scene: this.scene,
          bodies: this.bodies,
          mujocoRoot: this.mujocoRoot,
          requestReset: () => this.resetSimulation(),
          // Traced commands (ADR 0005 §3) run a graph per frame off these.
          rng: this.termRng,
          onnxSessions: this.policyGraphs,
          readOnnxSlot: this.readOnnxSlot,
        });
        this.commandManager.resetTerms();
        const motionTerm = this.commandManager.getTerm('motion');
        if (isMotionCommandTerm(motionTerm)) {
          await motionTerm.setSelectedMotion(
            config.motions?.find((motion) => motion.default)?.name
              ?? config.motions?.[0]?.name
              ?? null
          );
          motionTerm.setReferenceVisible?.(true);
        }
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        this.updateCachedState();
      }

      if (
        !(config.policy_num_actions as number | undefined) &&
        (!config.policy_joint_names || config.policy_joint_names.length === 0)
      ) {
        throw new Error('Policy config missing policy_joint_names.');
      }

      const runner = new PolicyRunner(config, {
        policyModules: {
          tracking: TrackingPolicy,
          locomotion: LocomotionPolicy,
        },
        observations: { ...Observations, ...this.policyPlugins.observations },
        // Expose the app-supplied clips to custom terms (they read bytes via
        // runner.getMotionData instead of fetching — ADR 0004 §4/§10).
        motions: policy.motions,
        // Traced observation terms (ADR 0005 §4): a graph each, fed from the sim.
        onnxSessions: this.policyGraphs,
        readOnnxSlot: this.readOnnxSlot,
      });

      await runner.init({
        mujoco: this.mujoco,
        mjModel: this.mjModel,
        mjData: this.mjData,
        scene: this.scene,
        commandManager: this.commandManager,
      });

      // Await observation preloads so clip-based obs don't return zeros on the first step.
      await runner.preloadAll();

      this.policyRunner = runner;
      this.policyStateBuilder = new PolicyStateBuilder(
        this.mujoco,
        this.mjModel,
        this.mjData,
        runner.getPolicyJointNames()
      );

      const state = this.policyStateBuilder.build();
      this.policyRunner.reset(state);
      this.policyControl = this.buildPolicyControl(config, runner, this.policyStateBuilder);

      // Infer decimation from fps in obs terms (default heuristic targets 0.02s, some tasks train finer).
      {
        const obsGroups = config.observations;
        const allTerms: Array<Record<string, unknown>> = [];
        if (Array.isArray(obsGroups)) {
          allTerms.push(...(obsGroups as Array<Record<string, unknown>>));
        } else if (obsGroups && typeof obsGroups === 'object') {
          for (const g of Object.values(obsGroups)) {
            if (Array.isArray(g)) allTerms.push(...(g as Array<Record<string, unknown>>));
          }
        }
        for (const term of allTerms) {
          if (!term || this.timestep <= 0) continue;
          if (typeof term.fps === 'number' && term.fps > 0) {
            this.policyCtrlDt = 1 / term.fps;
            const newDec = Math.max(1, Math.round(this.policyCtrlDt / this.timestep));
            if (newDec !== this.decimation) {
              console.log(`[PolicyRunner] Decimation updated: ${this.decimation} → ${newDec} (fps=${term.fps}, sim_dt=${this.timestep})`);
              this.decimation = newDec;
            }
            break;
          }
        }
      }

      // Initialize termination manager if termination config is present
      if (config.terminations && Object.keys(config.terminations).length > 0) {
        this.terminationManager = new TerminationManager(
          config.terminations,
          { ...Terminations, ...this.policyPlugins.terminations },
          runner,
          { onnxSessions: this.policyGraphs, readOnnxSlot: this.readOnnxSlot }
        );
        console.log(`[TerminationManager] ${this.terminationManager.size} termination term(s) loaded`);
      }

      const module = new OnnxModule(policy.onnx, config.onnx?.meta);
      await module.init();
      this.onnxModule = module;
      this.onnxInputDict = module.initInput();

      console.log('[PolicyRunner] config loaded', {
        obsSize: runner.getObservationSize(),
        obsLayout: runner.getObservationLayout(),
        pdEnabled: this.policyControl !== null,
      });
    } catch (error) {
      console.warn('Failed to load policy config:', error);
    }
  }

  private buildPolicyControl(
    config: PolicyConfig,
    runner: PolicyRunner,
    stateBuilder: PolicyStateBuilder
  ): Array<{
    controlType: string;
    ctrlAdr: number[];
    qposAdr: number[];
    qvelAdr: number[];
    actionIndices: number[];
    actionScale: Float32Array;
    actionOffset: Float32Array;
    defaultJointPos: Float32Array;
    encoderBias: Float32Array;
    positionActuator: boolean[];
    kp: Float32Array;
    kd: Float32Array;
    muscleNormalize: boolean;
  }> | null {
    const jointNames = runner.getPolicyJointNames();
    const affineBiasValue = this.mujoco.mjtBias?.mjBIAS_AFFINE?.value ?? 1;

    const buildEntry = (
      termKey: string,
      controlType: string,
      mapping: { ctrlAdr: number[]; qposAdr: number[]; qvelAdr: number[]; actionIndices: number[] },
      configScale: number[] | number | Record<string, number> | undefined,
      configOffset: number[] | number | Record<string, number> | undefined,
      configStiffness: number[] | number | Record<string, number> | undefined,
      configDamping: number[] | number | Record<string, number> | undefined,
      useDefaultOffset: boolean
    ) => {
      const n = mapping.qposAdr.length;
      const subsetJointNames = mapping.actionIndices.map((i) => jointNames[i]);

      const actionScale = this.normalizeControlArray(configScale, n, 1.0, subsetJointNames);
      const actionOffset = this.normalizeControlArray(configOffset, n, 0.0, subsetJointNames);

      const allDefaultJointPos = useDefaultOffset
        ? runner.getDefaultJointPos()
        : new Float32Array(jointNames.length);
      const defaultJointPos = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        defaultJointPos[i] = allDefaultJointPos[mapping.actionIndices[i]];
      }

      const allEncoderBias = Array.isArray(config.encoder_bias)
        ? Float32Array.from(config.encoder_bias)
        : new Float32Array(jointNames.length);
      const encoderBias = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        encoderBias[i] = allEncoderBias[mapping.actionIndices[i]] ?? 0.0;
      }

      const kp = this.normalizeControlArray(configStiffness, n, 0.0, subsetJointNames);
      const kd = this.normalizeControlArray(configDamping, n, 0.0, subsetJointNames);

      // Detect per-actuator whether the scene uses position actuators (biastype=affine,
      // ctrl=target_pos, PD handled internally by MuJoCo) or motor actuators
      // (biastype=none, ctrl=torque, PD must be computed externally from kp/kd).
      const positionActuator: boolean[] = mapping.ctrlAdr.map((adr) => {
        if (adr < 0 || !this.mjModel) return false;
        return this.mjModel.actuator_biastype[adr] === affineBiasValue;
      });

      const isPosition = positionActuator.some(Boolean);
      const isMotor = positionActuator.some((v) => !v);
      if (isPosition && isMotor) {
        console.warn(`[PolicyRunner] Action term "${termKey}": mixed actuator types detected.`);
      }
      console.log(
        `[PolicyRunner] Action term "${termKey}" (${controlType}): ${n} joint(s), ` +
        `mode: ${isPosition ? 'position (ctrl=target_pos)' : 'motor (ctrl=torque, external PD)'}`
      );

      return {
        controlType,
        ctrlAdr: mapping.ctrlAdr,
        qposAdr: mapping.qposAdr,
        qvelAdr: mapping.qvelAdr,
        actionIndices: mapping.actionIndices,
        actionScale,
        actionOffset,
        defaultJointPos,
        encoderBias,
        positionActuator,
        kp,
        kd,
        muscleNormalize: false,
      };
    };

    // ── Legacy path: no `actions` block, use flat top-level fields ──────────
    const actionsConfig = config.actions;
    if (!actionsConfig || Object.keys(actionsConfig).length === 0) {
      const controlType = config.control_type ?? 'joint_position';
      if (controlType !== 'joint_position' && controlType !== 'torque') {
        console.warn(`[PolicyRunner] Unsupported control_type: ${controlType}`);
        return null;
      }
      const baseMapping = stateBuilder.getControlMapping();
      if (!baseMapping) {
        console.warn('[PolicyRunner] Failed to build control mapping.');
        return null;
      }
      const mapping = {
        ...baseMapping,
        actionIndices: Array.from({ length: baseMapping.qposAdr.length }, (_, i) => i),
      };
      return [buildEntry(
        'legacy',
        controlType,
        mapping,
        config.action_scale,
        undefined,
        config.stiffness,
        config.damping,
        true
      )];
    }

    // ── Multi-term path: iterate every entry in the `actions` block ─────────
    const results: Array<ReturnType<typeof buildEntry>> = [];

    for (const [termKey, actionTerm] of Object.entries(actionsConfig)) {
      const controlType = actionTerm.type ?? 'joint_position';
      if (
        controlType !== 'joint_position' &&
        controlType !== 'torque' &&
        controlType !== 'muscle_activation'
      ) {
        console.warn(`[PolicyRunner] Action term "${termKey}": unsupported type "${controlType}", skipping.`);
        continue;
      }

      const patterns = actionTerm.actuator_names ?? ['.*'];

      if (controlType === 'muscle_activation') {
        const muscleMapping = stateBuilder.getCtrlMappingByActuatorNames(patterns);
        if (!muscleMapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": no actuators matched patterns [${patterns.join(', ')}], skipping.`);
          continue;
        }
        const n = muscleMapping.ctrlAdr.length;
        const actionScale = this.normalizeControlArray(
          actionTerm.scale as number[] | number | Record<string, number> | undefined,
          n,
          1.0
        );
        const actionOffset = this.normalizeControlArray(
          actionTerm.offset as number[] | number | Record<string, number> | undefined,
          n,
          0.0
        );
        const muscleNormalize = (actionTerm as { normalize?: boolean }).normalize ?? true;
        console.log(
          `[PolicyRunner] Action term "${termKey}" (muscle_activation): ${n} actuator(s), normalize=${muscleNormalize}`
        );
        results.push({
          controlType,
          ctrlAdr: muscleMapping.ctrlAdr,
          qposAdr: [],
          qvelAdr: [],
          actionIndices: muscleMapping.actionIndices,
          actionScale,
          actionOffset,
          defaultJointPos: new Float32Array(n),
          encoderBias: new Float32Array(n),
          positionActuator: new Array(n).fill(false),
          kp: new Float32Array(n),
          kd: new Float32Array(n),
          muscleNormalize,
        });
        continue;
      }

      // If actuator_names is absent or [".*"], match all joints (backward-compatible).
      const isMatchAll = patterns.length === 1 && patterns[0] === '.*';

      let mapping: { ctrlAdr: number[]; qposAdr: number[]; qvelAdr: number[]; actionIndices: number[] } | null;

      if (isMatchAll) {
        const baseMapping = stateBuilder.getControlMapping();
        if (!baseMapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": failed to build control mapping, skipping.`);
          continue;
        }
        mapping = {
          ...baseMapping,
          actionIndices: Array.from({ length: baseMapping.qposAdr.length }, (_, i) => i),
        };
      } else {
        mapping = stateBuilder.getControlMappingFor(patterns, jointNames);
        if (!mapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": no joints matched patterns [${patterns.join(', ')}], skipping.`);
          continue;
        }
      }

      const useDefaultOffset = actionTerm.use_default_offset !== undefined
        ? actionTerm.use_default_offset
        : controlType === 'joint_position';

      results.push(buildEntry(
        termKey,
        controlType,
        mapping,
        actionTerm.scale as number[] | number | Record<string, number> | undefined,
        actionTerm.offset as number[] | number | Record<string, number> | undefined,
        actionTerm.stiffness as number[] | number | Record<string, number> | undefined,
        actionTerm.damping as number[] | number | Record<string, number> | undefined,
        useDefaultOffset
      ));
    }

    if (results.length === 0) {
      console.warn('[PolicyRunner] No valid action terms found in config.actions.');
      return null;
    }
    return results;
  }

  private normalizeControlArray(
    values: number[] | number | Record<string, number> | undefined,
    length: number,
    fallback: number,
    jointNames?: string[]
  ): Float32Array {
    const output = new Float32Array(length);
    output.fill(fallback);
    if (typeof values === 'number') {
      output.fill(values);
      return output;
    }
    if (Array.isArray(values)) {
      for (let i = 0; i < length; i++) {
        output[i] = typeof values[i] === 'number' ? values[i] : fallback;
      }
      return output;
    }
    if (values !== null && typeof values === 'object' && jointNames) {
      for (const [name, val] of Object.entries(values)) {
        const idx = jointNames.indexOf(name);
        if (idx >= 0 && idx < length) {
          output[idx] = val;
        } else {
          console.warn(`[PolicyRunner] Joint name "${name}" not found in policy_joint_names; skipping.`);
        }
      }
      return output;
    }
    return output;
  }

  private resetSimulationState(): void {
    if (!this.mjModel || !this.mjData) {
      return;
    }
    if (this.mjModel.nkey > 0) {
      this.mujoco.mj_resetDataKeyframe(this.mjModel, this.mjData, 0);
    } else {
      this.mujoco.mj_resetData(this.mjModel, this.mjData);
    }
    if (this.initialQpos) {
      const qpos = this.mjData.qpos;
      for (let i = 0; i < Math.min(this.initialQpos.length, this.mjModel.nq); i++) {
        qpos[i] = this.initialQpos[i];
      }
    }
    if (this.initialQvel) {
      const qvel = this.mjData.qvel;
      for (let i = 0; i < Math.min(this.initialQvel.length, this.mjModel.nv); i++) {
        qvel[i] = this.initialQvel[i];
      }
    }
    // Not awaited: `resetSimulationState` is synchronous and so is the public
    // `resetSimulation()`. An ONNX reset term's write therefore lands a frame or
    // two late — the same accepted one-frame lag as the rest of the async
    // boundary (ADR 0005 §8), and the alternative is making reset async all the
    // way out through the engine API.
    void this.eventManager?.onReset(this.eventContext());
    this.commandManager.resetTerms();
    if (this.onnxModule) {
      this.onnxInputDict = this.onnxModule.initInput();
    }
    this.onnxTimeStep = 0;
    this.mujoco.mj_forward(this.mjModel, this.mjData);
    this.lastSimState.bodies.clear();
    this.updateCachedState();
  }

  private executeSimulationSteps(): void {
    if (!this.mjModel || !this.mjData) {
      return;
    }
    // Apply drag forces
    this.applyDragForces();

    for (let substep = 0; substep < this.decimation; substep++) {
      this.applyPolicyControl();
      this.mujoco.mj_step(this.mjModel, this.mjData);
    }
  }

  private applyPolicyControl(): void {
    if (!this.policyControl || !this.mjData) {
      return;
    }

    const ctrl = this.mjData.ctrl;
    ctrl.fill(0.0);

    // Fetch the full action vector once; each term reads its own slice via actionIndices.
    const allActions = this.policyRunner?.getLastActions() ?? new Float32Array(0);

    for (const term of this.policyControl) {
      const {
        controlType,
        ctrlAdr,
        qposAdr,
        qvelAdr,
        actionIndices,
        actionScale,
        actionOffset,
        defaultJointPos,
        encoderBias,
        positionActuator,
        kp,
        kd,
        muscleNormalize,
      } =
        term;
      const numJoints = ctrlAdr.length;

      if (controlType === 'joint_position') {
        for (let i = 0; i < numJoints; i++) {
          const ctrlIndex = ctrlAdr[i];
          if (ctrlIndex < 0) continue;
          const actionValue = allActions[actionIndices[i]] ?? 0;
          const target =
            defaultJointPos[i] +
            actionOffset[i] +
            actionScale[i] * actionValue -
            encoderBias[i];

          if (positionActuator[i]) {
            // Position actuator (biastype=affine): ctrl = target joint position.
            // MuJoCo computes force = kp*(ctrl - qpos) - kd*qvel internally.
            ctrl[ctrlIndex] = target;
          } else {
            // Motor actuator (biastype=none): ctrl = torque.
            // PD must be computed externally using kp/kd from the policy config.
            const qpos = this.mjData.qpos[qposAdr[i]];
            const qvel = this.mjData.qvel[qvelAdr[i]];
            ctrl[ctrlIndex] = kp[i] * (target - qpos) + kd[i] * (0 - qvel);
          }
        }
      } else if (controlType === 'torque') {
        for (let i = 0; i < numJoints; i++) {
          const ctrlIndex = ctrlAdr[i];
          if (ctrlIndex >= 0) {
            ctrl[ctrlIndex] = actionScale[i] * (allActions[actionIndices[i]] ?? 0);
          }
        }
      } else if (controlType === 'muscle_activation') {
        // Shared pre-step: raw = scale * action + offset.
        // normalize=true:  MyoSuite-canonical sigmoid σ(5 * (raw - 0.5)).
        // normalize=false: clip(raw, 0, 1) for models that already output excitation.
        for (let i = 0; i < numJoints; i++) {
          const ctrlIndex = ctrlAdr[i];
          if (ctrlIndex < 0) continue;
          const raw = (allActions[actionIndices[i]] ?? 0) * actionScale[i] + actionOffset[i];
          if (muscleNormalize) {
            ctrl[ctrlIndex] = 1 / (1 + Math.exp(-5 * (raw - 0.5)));
          } else {
            ctrl[ctrlIndex] = raw < 0 ? 0 : raw > 1 ? 1 : raw;
          }
        }
      }
    }
  }

  private async runOnnxInference(obs: Record<string, Float32Array>): Promise<void> {
    if (!this.onnxModule || !this.policyRunner || this.onnxInferencing) {
      return;
    }

    this.onnxInferencing = true;
    try {
      if (!this.onnxInputDict) {
        this.onnxInputDict = this.onnxModule.initInput();
      }
      const input: Record<string, ort.Tensor> = { ...this.onnxInputDict };
      if (this.onnxModule.inKeys.includes('time_step')) {
        input.time_step = new ort.Tensor('float32', new Float32Array([this.onnxTimeStep]), [1, 1]);
      }
      for (const [key, value] of Object.entries(obs)) {
        input[key] = new ort.Tensor('float32', value, [1, value.length]);
      }
      for (const key of this.onnxModule.inKeys) {
        if (!input[key]) {
          console.warn('[PolicyRunner] Missing ONNX input:', {
            key,
            available: Object.keys(input),
          });
          return;
        }
      }

      const [result, carry] = await this.onnxModule.runInference(input);
      if (Object.keys(carry).length > 0) {
        this.onnxInputDict = { ...this.onnxInputDict, ...carry };
      }
      if (this.onnxModule.inKeys.includes('time_step')) {
        this.onnxTimeStep += 1;
      }

      const outKey = this.onnxModule.outKeys[0];
      const actionTensor = result.action ?? (outKey ? result[outKey] : null) ?? result.policy ?? null;
      if (!actionTensor) {
        return;
      }

      const raw = actionTensor.data as Float32Array | number[];
      const action = ArrayBuffer.isView(raw) ? new Float32Array(raw) : Float32Array.from(raw);
      const expectedActionCount = this.policyRunner?.getNumActions() ?? 0;
      if (this.policyControl && action.length !== expectedActionCount) {
        console.warn('[PolicyRunner] Action size mismatch:', {
          expected: expectedActionCount,
          got: action.length,
        });
        return;
      }
      this.policyRunner.setLastActions(action);
    } catch (error) {
      console.warn('[PolicyRunner] ONNX inference failed:', error);
    } finally {
      this.onnxInferencing = false;
    }
  }

  private applyDragForces(): void {
    if (!this.dragStateManager || !this.mjModel || !this.mjData || !this.bodies) {
      return;
    }

    // Clear xfrc_applied (reset to zero at each step)
    for (let i = 0; i < this.mjData.xfrc_applied.length; i++) {
      this.mjData.xfrc_applied[i] = 0.0;
    }

    const dragged = this.dragStateManager.physicsObject;
    if (!dragged || !('bodyID' in dragged) || typeof dragged.bodyID !== 'number' || dragged.bodyID <= 0) {
      return;
    }

    const bodyId = dragged.bodyID as number;
    if (this.dynamicBodyIds && !this.dynamicBodyIds.has(bodyId)) {
      return;
    }

    // Update body positions (for drag calculation)
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (this.bodies[b]) {
        getPosition(this.mjData.xpos, b, this.bodies[b].position);
        getQuaternion(this.mjData.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix(true, false);
      }
    }

    // Update offset
    this.dragStateManager.update();

    // Calculate force (Three.js coordinate system → MuJoCo coordinate system)
    const forceThree = this.dragStateManager.offset
      .clone()
      .multiplyScalar(this.dragForceScale);
    const force = threeToMjcCoordinate(forceThree);

    // Point where force is applied (world coordinates)
    const pointThree = this.dragStateManager.worldHit.clone();
    const point = threeToMjcCoordinate(pointThree);
    // Body position
    const bodyPos = new THREE.Vector3(
      this.mjData.xpos[bodyId * 3 + 0],
      this.mjData.xpos[bodyId * 3 + 1],
      this.mjData.xpos[bodyId * 3 + 2]
    );

    // Calculate torque: τ = r × F
    const r = new THREE.Vector3(
      point.x - bodyPos.x,
      point.y - bodyPos.y,
      point.z - bodyPos.z
    );
    const f = new THREE.Vector3(force.x, force.y, force.z);
    const torque = new THREE.Vector3().crossVectors(r, f);

    // Set xfrc_applied
    // xfrc_applied: (nbody, 6) = [fx, fy, fz, tx, ty, tz] for each body
    const offset = bodyId * 6;
    this.mjData.xfrc_applied[offset + 0] = force.x;
    this.mjData.xfrc_applied[offset + 1] = force.y;
    this.mjData.xfrc_applied[offset + 2] = force.z;
    this.mjData.xfrc_applied[offset + 3] = torque.x;
    this.mjData.xfrc_applied[offset + 4] = torque.y;
    this.mjData.xfrc_applied[offset + 5] = torque.z;
  }

  private updateCachedState(): void {
    if (!this.mjModel || !this.mjData || !this.bodies) {
      return;
    }
    const dynamicBodyIds = this.dynamicBodyIds;
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (dynamicBodyIds && !dynamicBodyIds.has(b)) {
        continue;
      }
      if (this.bodies[b]) {
        if (!this.lastSimState.bodies.has(b)) {
          this.lastSimState.bodies.set(b, {
            position: new THREE.Vector3(),
            quaternion: new THREE.Quaternion(),
          });
        }
        const state = this.lastSimState.bodies.get(b) as BodyState;
        getPosition(this.mjData.xpos, b, state.position);
        getQuaternion(this.mjData.xquat, b, state.quaternion);
      }
    }

    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      updateTendonGeometry(
        this.mjModel,
        this.mjData,
        {
          cylinders: this.mujocoRoot.cylinders,
          spheres: this.mujocoRoot.spheres!,
        },
        this.lastSimState.tendons
      );
    }
  }

  private applyViewerConfig(config: ViewerConfig | null): void {
    this.cameraState = applyViewerConfig(config, this.camera, this.controls, this.mjModel, this.mjData);
  }

  private computeDynamicBodyIds(mjModel: MjModel): Set<number> {
    const dynamic = new Set<number>();
    for (let bodyId = 1; bodyId < mjModel.nbody; bodyId++) {
      let current = bodyId;
      while (current > 0) {
        if (mjModel.body_jntnum[current] > 0) {
          dynamic.add(bodyId);
          break;
        }
        current = mjModel.body_parentid[current];
      }
    }
    return dynamic;
  }

  private syncStaticBodiesFromData(): void {
    if (!this.mjModel || !this.mjData || !this.bodies) {
      return;
    }
    const dynamicBodyIds = this.dynamicBodyIds;
    for (let bodyId = 0; bodyId < this.mjModel.nbody; bodyId++) {
      if (dynamicBodyIds?.has(bodyId)) {
        continue;
      }
      const body = this.bodies[bodyId];
      if (!body) {
        continue;
      }
      getPosition(this.mjData.xpos, bodyId, body.position);
      getQuaternion(this.mjData.xquat, bodyId, body.quaternion);
    }
  }

  private render = (): void => {
    this.commandManager.updateDebugVisuals();

    if (this.mjData) {
      updateCameraFromData(this.mjData, this.camera, this.controls, this.cameraState);
    }
    this.controls.update();

    if (this.mjModel && this.mjData && this.bodies) {
      updateHeadlightFromCamera(this.camera, this.lights);

      for (const [b, state] of this.lastSimState.bodies) {
        const body = this.bodies[b];
        if (body) {
          body.position.copy(state.position);
          body.quaternion.copy(state.quaternion);
        }
      }

      updateLightsFromData(this.mujoco, this.mjData, this.lights);

      if (this.mujocoRoot && this.mujocoRoot.cylinders) {
        updateTendonRendering(
          {
            cylinders: this.mujocoRoot.cylinders,
            spheres: this.mujocoRoot.spheres!,
          },
          this.lastSimState.tendons
        );
      }
    }

    this.renderer.render(this.scene, this.camera);
  };

  /**
   * Capture the current view as a JPEG Blob. Renders one frame and immediately
   * copies it into a 2D canvas in the same synchronous turn, so it works without
   * `preserveDrawingBuffer` — the WebGL drawing buffer is still intact before the
   * browser composites. Used by the upload preview's "Scan thumbnail" (ADR 0005).
   */
  async captureThumbnail(options: { maxDim?: number; quality?: number } = {}): Promise<Blob> {
    const { maxDim = 1280, quality = 0.85 } = options;
    const src = this.renderer.domElement;

    this.renderer.render(this.scene, this.camera);

    const sw = src.width;
    const sh = src.height;
    if (!sw || !sh) {
      throw new Error('mjswan: canvas has zero size; nothing to capture.');
    }

    const scale = Math.min(1, maxDim / Math.max(sw, sh));
    const dw = Math.max(1, Math.round(sw * scale));
    const dh = Math.max(1, Math.round(sh * scale));

    const out = document.createElement('canvas');
    out.width = dw;
    out.height = dh;
    const ctx = out.getContext('2d');
    if (!ctx) {
      throw new Error('mjswan: failed to get a 2D context for thumbnail capture.');
    }
    ctx.drawImage(src, 0, 0, dw, dh);

    return await new Promise<Blob>((resolve, reject) => {
      out.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error('mjswan: toBlob returned null.'))),
        'image/jpeg',
        quality
      );
    });
  }

  private onWindowResize = (): void => {
    const { width, height } = this.getSize();
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  };

  async dispose(): Promise<void> {
    // Await stop so the loop halts before we free the state it reads.
    await this.stop();
    this.policyRunner = null;
    this.policyStateBuilder = null;

    // ONNX session, splat, and collider are not cache-managed; free them here
    // or they leak across navigations (a live splat worker freezes the tab).
    if (this.onnxModule) {
      this.onnxModule.dispose();
      this.onnxModule = null;
    }
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }

    if (this.dragStateManager) {
      this.dragStateManager.dispose();
      this.dragStateManager = null;
    }

    this.mjData = null;
    this.mjModel = null;

    // NOTE: Do NOT dispose Three.js resources here as they may be cached
    // The cache manager will handle their disposal when evicting
    // Just clear references
    // this.disposeThreeJSResources();

    window.removeEventListener('resize', this.onWindowResize);
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;

    this.controls.dispose();
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();

    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }

    if (this.vrButton?.parentElement) {
      this.vrButton.parentElement.removeChild(this.vrButton);
      this.vrButton = null;
    }

    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.dynamicBodyIds = null;
    this.lastSimState.bodies.clear();
    this.commandManager.dispose();
  }

  private disposeThreeJSResources(): void {
    if (!this.scene) {
      return;
    }

    this.scene.traverse((object) => {
      if ('geometry' in object && object.geometry) {
        (object.geometry as THREE.BufferGeometry).dispose();
      }
      if ('material' in object && object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach((material) => this.disposeMaterial(material));
        } else {
          this.disposeMaterial(object.material as THREE.Material);
        }
      }
    });

    while (this.scene.children.length > 0) {
      this.scene.remove(this.scene.children[0]);
    }
  }

  private disposeMaterial(material: THREE.Material): void {
    const anyMaterial = material as THREE.MeshStandardMaterial & {
      map?: THREE.Texture;
      aoMap?: THREE.Texture;
      emissiveMap?: THREE.Texture;
      metalnessMap?: THREE.Texture;
      normalMap?: THREE.Texture;
      roughnessMap?: THREE.Texture;
    };

    if (anyMaterial.map) {
      anyMaterial.map.dispose();
    }
    if (anyMaterial.aoMap) {
      anyMaterial.aoMap.dispose();
    }
    if (anyMaterial.emissiveMap) {
      anyMaterial.emissiveMap.dispose();
    }
    if (anyMaterial.metalnessMap) {
      anyMaterial.metalnessMap.dispose();
    }
    if (anyMaterial.normalMap) {
      anyMaterial.normalMap.dispose();
    }
    if (anyMaterial.roughnessMap) {
      anyMaterial.roughnessMap.dispose();
    }
    material.dispose();
  }

  private getSize(): { width: number; height: number } {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;
    return {
      width: Math.max(1, width),
      height: Math.max(1, height),
    };
  }
}
