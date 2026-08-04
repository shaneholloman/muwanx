import { ObservationBase } from '../observation/ObservationBase';
import {
  FusedObservation,
  isFusedObservationConfig,
  type FusedObservationConfig,
} from '../observation/FusedObservation';
import { HistoryObservation, historyOffsets } from '../observation/HistoryObservation';
import {
  NativeObservation,
  isNativeObservationConfig,
} from '../observation/NativeObservation';
import {
  OnnxObservation,
  isOnnxObservationConfig,
  type OnnxObservationConfig,
} from '../observation/OnnxObservation';
import type { OnnxSessionCache, SlotReader } from '../onnx/session';
import { type Bytes, resolveBytes } from '../utils/bytes';
import { PolicyModule } from './PolicyModule';
import type {
  ObservationConfigEntry,
  PolicyConfig,
  PolicyRunnerContext,
  PolicyState,
} from './types';

export type PolicyModuleConstructor = new (config: PolicyConfig) => PolicyModule;
export type ObservationConstructor = new (
  runner: PolicyRunner,
  config: ObservationConfigEntry
) => ObservationBase;

export type PolicyRunnerOptions = {
  policyModules?: Record<string, PolicyModuleConstructor>;
  observations?: Record<string, ObservationConstructor>;
  /** App-supplied motion clips (name → bytes), exposed to terms via getMotionData. */
  motions?: Array<{ name: string; data: Bytes }>;
  /**
   * Deps for ONNX-backed observation terms (ADR 0005): the loaded `.onnx`
   * sessions keyed by the path each entry's `onnx` field names, and the reader
   * for the dynamic state slots those graphs declare. Absent for a policy with
   * no traced observations.
   */
  onnxSessions?: OnnxSessionCache;
  readOnnxSlot?: SlotReader;
};

export class PolicyRunner {
  private config: PolicyConfig;
  private options: PolicyRunnerOptions;
  private policyModule: PolicyModule | null;
  private obsGroups: Record<string, ObservationBase[]>;
  private obsLayouts: Record<string, { name: string; size: number }[]>;
  private obsSizes: Record<string, number>;
  private historyConfig: Record<string, { steps: number; interleaved: boolean }>;
  private historyBuffers: Record<string, Float32Array>;
  /** Groups whose history must be filled with the next frame (set by `reset()`). */
  private historyNeedsPrime: Record<string, boolean> = {};
  private defaultObsKey: string | null;
  private context: PolicyRunnerContext | null;
  private policyJointNames: string[];
  private defaultJointPos: Float32Array;
  private encoderBias: Float32Array;
  private numActions: number;
  private lastActions: Float32Array;
  private motionCache: Map<string, Promise<ArrayBuffer | null>> = new Map();

  constructor(config: PolicyConfig, options: PolicyRunnerOptions = {}) {
    this.config = config;
    this.options = options;
    this.policyModule = null;
    this.obsGroups = {};
    this.obsLayouts = {};
    this.obsSizes = {};
    this.historyConfig = {};
    this.historyBuffers = {};
    this.historyNeedsPrime = {};
    this.defaultObsKey = null;
    this.context = null;

    this.policyJointNames = (config.policy_joint_names ?? []).slice();
    this.numActions = (config.policy_num_actions as number | undefined) ?? this.policyJointNames.length;
    this.lastActions = new Float32Array(this.numActions);
    this.defaultJointPos = this.normalizeArray(
      config.default_joint_pos ?? [],
      this.numActions,
      0.0
    );
    this.encoderBias = this.normalizeArray(
      config.encoder_bias ?? [],
      this.numActions,
      0.0
    );
  }

  async init(context: PolicyRunnerContext): Promise<void> {
    this.context = context;
    this.policyModule = await this.buildPolicyModule(context);
    this.buildObservationGroups();
  }

  reset(state?: PolicyState): void {
    this.lastActions.fill(0.0);
    this.policyModule?.reset();
    for (const obsList of Object.values(this.obsGroups)) {
      for (const obs of obsList) {
        if (obs.reset) {
          obs.reset(state);
        }
      }
    }
    if (state) {
      // Priming a history buffer means computing a frame, which is async once a
      // term is ONNX-backed (ADR 0005) — while `reset()` is called from the
      // engine's synchronous public `resetSimulation()`. So flag it and prime on
      // the next collect instead, filling every slot with the frame that is
      // actually about to be used rather than a separately-computed one.
      for (const [key, config] of Object.entries(this.historyConfig)) {
        if (config.steps > 1) this.historyNeedsPrime[key] = true;
      }
    }
  }

  update(state: PolicyState): void {
    this.policyModule?.update();
    for (const obsList of Object.values(this.obsGroups)) {
      for (const obs of obsList) {
        if (obs.update) {
          obs.update(state);
        }
      }
    }
  }

  /** Async because ONNX-backed terms run ORT inference (ADR 0005 §8). */
  async collectObservationsByKey(state: PolicyState): Promise<Record<string, Float32Array>> {
    this.update(state);
    const outputs: Record<string, Float32Array> = {};

    for (const [key, obsList] of Object.entries(this.obsGroups)) {
      const history = this.historyConfig[key];
      if (history && history.steps > 1) {
        const frame = await this.buildFrame(obsList, state);
        const buffer = this.historyBuffers[key];
        if (this.historyNeedsPrime[key]) {
          // First frame after a reset: every slot is this frame, so the policy
          // never sees a history of zeros it was not trained on.
          for (let i = 0; i < history.steps; i++) buffer.set(frame, i * frame.length);
          delete this.historyNeedsPrime[key];
        } else {
          for (let i = buffer.length - 1; i >= frame.length; i--) {
            buffer[i] = buffer[i - frame.length];
          }
          buffer.set(frame, 0);
        }
        outputs[key] = new Float32Array(buffer);
      } else {
        outputs[key] = await this.buildFrame(obsList, state);
      }
    }
    return outputs;
  }

  async collectObservations(state: PolicyState): Promise<Float32Array> {
    const outputs = await this.collectObservationsByKey(state);
    if (this.defaultObsKey && outputs[this.defaultObsKey]) {
      return outputs[this.defaultObsKey];
    }
    const first = Object.keys(outputs)[0];
    return first ? outputs[first] : new Float32Array(0);
  }

  /** Await all observation preload() promises before the first inference step. */
  async preloadAll(): Promise<void> {
    const promises: Promise<void>[] = [];
    for (const obsList of Object.values(this.obsGroups)) {
      for (const obs of obsList) {
        if (typeof obs.preload === 'function') {
          promises.push(obs.preload());
        }
      }
    }
    await Promise.all(promises);
  }

  getObservationSize(): number {
    if (this.defaultObsKey && this.obsSizes[this.defaultObsKey] !== undefined) {
      return this.obsSizes[this.defaultObsKey];
    }
    const first = Object.keys(this.obsSizes)[0];
    return first ? this.obsSizes[first] : 0;
  }

  getObservationLayout(): { name: string; size: number }[] {
    if (this.defaultObsKey && this.obsLayouts[this.defaultObsKey]) {
      return this.obsLayouts[this.defaultObsKey].map((entry) => ({ ...entry }));
    }
    const first = Object.keys(this.obsLayouts)[0];
    return first ? this.obsLayouts[first].map((entry) => ({ ...entry })) : [];
  }

  getPolicyModuleContext(): Record<string, unknown> {
    return this.policyModule?.getContext() ?? {};
  }

  getPolicyModule(): PolicyModule | null {
    return this.policyModule;
  }

  getContext(): PolicyRunnerContext | null {
    return this.context;
  }

  getPolicyJointNames(): string[] {
    return this.policyJointNames.slice();
  }

  getNumActions(): number {
    return this.numActions;
  }

  getDefaultJointPos(): Float32Array {
    return new Float32Array(this.defaultJointPos);
  }

  getEncoderBias(): Float32Array {
    return new Float32Array(this.encoderBias);
  }

  getLastActions(): Float32Array {
    return new Float32Array(this.lastActions);
  }

  getConfig(): PolicyConfig {
    return this.config;
  }

  /**
   * Resolve an app-supplied motion clip's raw bytes by name (cached), or null
   * if not supplied. Custom terms that need clip data read this slot instead of
   * fetching a URL — the app owns and feeds all bytes (ADR 0004 §4/§10).
   */
  getMotionData(name: string): Promise<ArrayBuffer | null> {
    const cached = this.motionCache.get(name);
    if (cached) return cached;
    const motion = this.options.motions?.find((m) => m.name === name);
    const promise: Promise<ArrayBuffer | null> = motion
      ? resolveBytes(motion.data)
      : Promise.resolve(null);
    this.motionCache.set(name, promise);
    return promise;
  }

  setLastActions(actions: Float32Array): void {
    if (actions.length !== this.lastActions.length) {
      this.lastActions = new Float32Array(actions);
      return;
    }
    this.lastActions.set(actions);
  }

  private async buildPolicyModule(
    context: PolicyRunnerContext
  ): Promise<PolicyModule | null> {
    const registry = this.options.policyModules ?? {};
    const moduleKey = this.config.policy_module;
    const Module = moduleKey ? registry[moduleKey] : registry.default;

    if (moduleKey && !Module) {
      throw new Error(`Unknown policy module: ${moduleKey}`);
    }

    if (!Module) {
      return null;
    }

    const module = new Module(this.config);
    await module.init(context);
    return module;
  }

  private buildObservationGroups(): void {
    const registry = this.options.observations ?? {};
    const obsConfig = this.config.observations ?? {};
    this.obsGroups = {};
    this.obsLayouts = {};
    this.obsSizes = {};
    this.historyConfig = {};
    this.historyBuffers = {};
    this.historyNeedsPrime = {};
    this.defaultObsKey = null;

    const buildTerm = (entry: ObservationConfigEntry): ObservationBase => {
      // ONNX-traced and natively-computed terms (ADR 0005) bypass the class
      // registry: they are one generic handler each, configured entirely by data,
      // so `entry.name` is the term's own identity rather than a class to look up.
      if (isOnnxObservationConfig(entry)) {
        return this.buildOnnxObservation(entry);
      }
      if (isNativeObservationConfig(entry)) {
        return new NativeObservation(this, entry);
      }
      const ObsClass = registry[entry.name];
      if (!ObsClass) {
        throw new Error(`Unknown observation type: ${entry.name}`);
      }
      return new ObsClass(this, entry);
    };

    // Per-term history wraps the term (mjlab stacks per term, before concatenating);
    // the legacy registry classes take `history_steps` themselves, so only the
    // ONNX/native entries the build emits `history_length`/`history_offsets` for are
    // wrapped here.
    const buildObservation = (entry: ObservationConfigEntry): ObservationBase => {
      const base = buildTerm(entry);
      const offsets =
        isOnnxObservationConfig(entry) || isNativeObservationConfig(entry)
          ? historyOffsets(entry)
          : null;
      return offsets ? new HistoryObservation(this, entry, base, offsets) : base;
    };

    for (const [key, value] of Object.entries(obsConfig)) {
      // A fused group (ADR 0005 §4) is one graph for all its terms; the per-term
      // list below is what a group that could not fuse still uses.
      if (isFusedObservationConfig(value)) {
        const fused = this.buildFusedObservation(key, value);
        this.registerGroup(key, [fused], [{ name: key }], undefined, value.layout);
        continue;
      }
      if (Array.isArray(value)) {
        const obsList = value.map(buildObservation);
        this.registerGroup(key, obsList, value);
        continue;
      }
      if (value && typeof value === 'object') {
        const configValue = value as {
          history_steps?: number;
          interleaved?: boolean;
          components?: ObservationConfigEntry[];
        };
        if (Array.isArray(configValue.components)) {
          // Group-level history owns the stacking here, so each component computes a
          // single frame. Same builder as the array shape, so this group form gets
          // ONNX/native terms too rather than only registry classes.
          const obsList = configValue.components.map((entry) =>
            buildObservation({ ...entry, history_steps: 1 }),
          );
          const steps = Math.max(1, Math.floor(configValue.history_steps ?? 1));
          const interleaved = Boolean(configValue.interleaved);
          this.registerGroup(key, obsList, configValue.components, {
            steps,
            interleaved,
          });
        }
      }
    }

    if (this.obsGroups.policy) {
      this.defaultObsKey = 'policy';
    } else if (this.obsGroups.observation) {
      this.defaultObsKey = 'observation';
    } else if (this.obsGroups.obs_history) {
      this.defaultObsKey = 'obs_history';
    } else {
      this.defaultObsKey = Object.keys(this.obsGroups)[0] ?? null;
    }
  }

  /**
   * Build a traced-ONNX observation term, or throw if its deps are missing.
   *
   * Unlike `OnnxCommand`/`OnnxEvent` — which warn and skip, so one absent session
   * cannot take down a whole scene — an observation is part of the policy's input
   * vector. Dropping it would silently shift every later term's offset and feed
   * the network a differently-shaped observation, so this fails loudly instead.
   */
  private buildOnnxObservation(entry: OnnxObservationConfig): OnnxObservation {
    const session = this.options.onnxSessions?.get(entry.onnx);
    const readSlot = this.options.readOnnxSlot;
    if (!session || !readSlot) {
      throw new Error(
        `Observation "${entry.name}" needs the ONNX session "${entry.onnx}" and a ` +
          'slot reader; pass onnxSessions/readOnnxSlot in PolicyRunnerOptions.'
      );
    }
    return new OnnxObservation(this, entry, { session, readSlot });
  }

  /**
   * Build the single handler for a fused group, or throw.
   *
   * Loud like the per-term case, and for the same reason: a group with no graph is
   * a policy with no input vector, not a degraded one.
   */
  private buildFusedObservation(
    key: string,
    config: FusedObservationConfig
  ): FusedObservation {
    const session = this.options.onnxSessions?.get(config.fused);
    const readSlot = this.options.readOnnxSlot;
    if (!session || !readSlot) {
      throw new Error(
        `Observation group "${key}" needs the ONNX session "${config.fused}" and a ` +
          'slot reader; pass onnxSessions/readOnnxSlot in PolicyRunnerOptions.'
      );
    }
    return new FusedObservation(this, { ...config, name: key }, { session, readSlot });
  }

  private registerGroup(
    key: string,
    obsList: ObservationBase[],
    configList: ObservationConfigEntry[],
    history?: { steps: number; interleaved: boolean },
    /** Fused groups only: per-term widths, since one handler covers every term. */
    fusedLayout?: Array<{ name: string; size: number }>
  ): void {
    this.obsGroups[key] = obsList;
    this.obsLayouts[key] = fusedLayout
      ? fusedLayout.map((entry) => ({ ...entry }))
      : obsList.map((obs, index) => ({
          name: configList[index]?.name ?? `obs_${index}`,
          size: obs.size,
        }));
    const baseSize = this.obsLayouts[key].reduce((sum, entry) => sum + entry.size, 0);
    if (history && history.steps > 1) {
      this.historyConfig[key] = history;
      this.historyBuffers[key] = new Float32Array(baseSize * history.steps);
      this.obsSizes[key] = baseSize * history.steps;
    } else {
      this.obsSizes[key] = baseSize;
    }
  }

  private async buildFrame(
    obsList: ObservationBase[],
    state: PolicyState
  ): Promise<Float32Array> {
    // Compute every term first, then size the buffer from the actual arrays, so
    // an observation whose output length changes between frames can never
    // overflow `set()` (a term's `size` getter may lag its output — e.g. it is
    // cached from the previous frame). The guard keeps a clear error for a
    // genuine size mismatch.
    //
    // Terms are kicked off together and awaited as a batch: an ONNX-backed term
    // (ADR 0005) runs async ORT inference, and awaiting them one at a time would
    // serialize the group's graphs for no reason.
    const arrays = await Promise.all(
      obsList.map(async (obs) => {
        const value = await obs.compute(state);
        const array = value instanceof Float32Array ? value : Float32Array.from(value);
        if (array.length !== obs.size) {
          throw new Error(
            `Observation size mismatch: expected ${obs.size}, got ${array.length}`
          );
        }
        return array;
      })
    );
    const total = arrays.reduce((sum, array) => sum + array.length, 0);
    const output = new Float32Array(total);
    let offset = 0;
    for (const array of arrays) {
      output.set(array, offset);
      offset += array.length;
    }
    return output;
  }

  private normalizeArray(
    values: number[],
    length: number,
    fallback: number
  ): Float32Array {
    const output = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      output[i] = typeof values[i] === 'number' ? values[i] : fallback;
    }
    return output;
  }
}
