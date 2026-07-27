/**
 * Shared ONNX Runtime Web session abstraction (ADR 0005).
 *
 * `OnnxCommand` and `OnnxEvent` both need to run small term-body graphs; this is
 * the one place that talks to `onnxruntime-web` so the two don't duplicate it.
 * The `OnnxSession`/`OnnxTensorLike` interfaces are intentionally minimal (one
 * async method) so command/event handlers stay testable with a fake — no ORT,
 * no browser, no WASM required for their own unit tests.
 */
import * as ort from 'onnxruntime-web';

/** Minimal ORT-Web surface a command/event handler needs. */
export interface OnnxSession {
  run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>>;
}

export interface OnnxTensorLike {
  data: Float32Array | BigInt64Array | Uint8Array;
  dims: readonly number[];
}

/**
 * A dynamic runtime read a term's graph declares as an input.
 *
 * Three shapes, distinguished by which field is set (mirroring
 * `mjswan.compile.tracer.slot_to_json`):
 * - `entity` + `field` — one `Entity.data.<field>` tensor.
 * - `sensor` — a whole MuJoCo sensor's value (mjlab's `builtin_sensor`).
 * - `command` + `field` — another command term's current state, e.g. a goal
 *   position an observation measures distance to (mjlab's
 *   `object_to_goal_distance`).
 *
 * `input` is the graph input name to feed this slot's value as. Prefer it over
 * re-deriving a name from `entity`/`field`: sensor and command names carry
 * paths/dots that the build folds to identifiers, so the mapping is not
 * reproducible here. Optional only for backward compatibility with configs
 * emitted before it existed — see `slotInputName`.
 *
 * `shape` is the traced tensor's shape, batch axis included. A slot reader hands
 * back a flat array, so the rank has to travel with the slot — see `slotDims`.
 */
export interface OnnxInputSlot {
  entity?: string | null;
  field?: string;
  sensor?: string;
  command?: string;
  input?: string;
  shape?: number[];
}

/** The graph input name for a slot: the build-supplied one, else the legacy scheme. */
export function slotInputName(slot: OnnxInputSlot): string {
  if (slot.input) return slot.input;
  return `${slot.entity ?? 'entity'}__${slot.field ?? ''}`;
}

/**
 * The dims to feed a slot's flat value as.
 *
 * Not every `Entity.data` field is rank 2: `site_pos_w` is
 * `(batch, num_sites, 3)` and `heading_w` is `(batch,)`, and ORT rejects a rank
 * mismatch outright rather than reshaping. So the traced shape is authoritative,
 * with the batch axis pinned to 1 (the browser runs a single env). `[1, length]`
 * is the fallback for a slot from a build that predates `shape`, and also the
 * repair when the declared element count no longer matches what the model
 * actually has — feeding the declared shape there would be a lie about the data.
 */
export function slotDims(slot: OnnxInputSlot, length: number): number[] {
  const shape = slot.shape;
  if (!shape || shape.length === 0) return [1, length];
  const declared = shape.slice(1).reduce((a, b) => a * b, 1);
  if (declared !== length) return [1, length];
  return [1, ...shape.slice(1)];
}

/** Reads the dynamic runtime state an `OnnxInputSlot` declares, or null if absent. */
export type SlotReader = (slot: OnnxInputSlot) => Float32Array | null;

function toOrtTensor(t: OnnxTensorLike): ort.Tensor {
  if (t.data instanceof Uint8Array) {
    // Uint8Array carries our bool convention (0/1); ORT's 'bool' dtype expects
    // a Uint8Array of the same length, so this is a direct pass-through.
    return new ort.Tensor('bool', t.data, t.dims);
  }
  if (t.data instanceof BigInt64Array) {
    return new ort.Tensor('int64', t.data, t.dims);
  }
  return new ort.Tensor('float32', t.data, t.dims);
}

function fromOrtTensor(t: ort.Tensor): OnnxTensorLike {
  if (t.type === 'bool') return { data: t.data as Uint8Array, dims: t.dims };
  if (t.type === 'int64') return { data: t.data as BigInt64Array, dims: t.dims };
  return { data: t.data as Float32Array, dims: t.dims };
}

/** Wraps a real ORT-Web `InferenceSession` behind the minimal `OnnxSession` shape. */
class OrtSession implements OnnxSession {
  constructor(private readonly session: ort.InferenceSession) {}

  async run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>> {
    const ortFeeds: Record<string, ort.Tensor> = {};
    for (const [name, tensor] of Object.entries(feeds)) ortFeeds[name] = toOrtTensor(tensor);
    const outputs = await this.session.run(ortFeeds);
    const result: Record<string, OnnxTensorLike> = {};
    for (const [name, tensor] of Object.entries(outputs)) result[name] = fromOrtTensor(tensor);
    return result;
  }
}

/** Create a real ORT-Web-backed session from graph bytes (never fetches — ADR 0004 §4). */
export async function createOnnxSession(bytes: ArrayBuffer): Promise<OnnxSession> {
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  return new OrtSession(session);
}

/**
 * A named collection of small term-body graphs (command or event onnx assets),
 * built once from resolved bytes and looked up by their `policy.json`/
 * `config.json` path (e.g. `"command/twist.onnx"`, `"event/push_robot.onnx"`).
 *
 * `sessionFactory` is injectable so callers can build the cache with a fake in
 * tests; production code omits it and gets `createOnnxSession`.
 */
export class OnnxSessionCache {
  private sessions = new Map<string, OnnxSession>();

  constructor(
    private readonly sessionFactory: (bytes: ArrayBuffer) => Promise<OnnxSession> = createOnnxSession,
  ) {}

  /** Build (or replace) sessions for the given `{name, data}` entries. */
  async load(entries: ReadonlyArray<{ name: string; data: ArrayBuffer }>): Promise<void> {
    await Promise.all(
      entries.map(async (entry) => {
        this.sessions.set(entry.name, await this.sessionFactory(entry.data));
      }),
    );
  }

  get(path: string): OnnxSession | undefined {
    return this.sessions.get(path);
  }

  get size(): number {
    return this.sessions.size;
  }

  clear(): void {
    this.sessions.clear();
  }
}
