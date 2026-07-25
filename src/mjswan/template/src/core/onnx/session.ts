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

/** A dynamic runtime read a command/event term's graph declares as an input. */
export interface OnnxInputSlot {
  entity?: string | null;
  field: string;
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
