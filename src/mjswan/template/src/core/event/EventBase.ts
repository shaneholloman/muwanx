export type EventConfig = {
  name: string;
  params?: Record<string, unknown>;
  /**
   * ONNX-backed events (ADR 0005 §3/§4) carry these; a plain `EventConfig` (no
   * `mode`/`onnx`) is a legacy reset-only registry/DslEvent term, unchanged.
   * See `OnnxEventConfig` in `./OnnxEvent` for the authoritative shape.
   */
  mode?: 'startup' | 'reset' | 'interval';
  onnx?: string;
  rand_dim?: number;
  input_slots?: Array<{ entity?: string | null; field: string }>;
  write_targets?: unknown[];
  rand_ranges?: Array<[number, number]>;
  interval_range_s?: [number, number];
  is_global_time?: boolean;
  min_step_count_between_reset?: number;
};

export type EventContext = {
  mjModel: import('mujoco').MjModel | null;
  mjData: import('mujoco').MjData | null;
  terrainData?: TerrainData | null;
};

export type TerrainData = {
  flat_patches?: Record<string, number[][]>;
};

export abstract class EventBase {
  protected config: EventConfig;

  constructor(config: EventConfig) {
    this.config = config;
  }

  /** Called on every episode reset. */
  abstract onReset(context: EventContext): void;
}

export type EventConstructor = new (config: EventConfig) => EventBase;
