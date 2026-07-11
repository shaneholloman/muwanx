# `createEngine`: a headless, instance-scoped engine core with an app-owned catalog

> Status: **Proposed**. A large refactor with **no backward-compatibility
> constraint** (pre-1.0; versioning is managed out-of-band). This replaces an
> earlier draft of this ADR that proposed an additive `controls` surface on the
> existing React `mount()`; that additive approach is superseded by the
> engine/app split below.

## Context

The library build (`dist/mjswan.js`) is a React app: `mount()` → `MountApp.tsx` →
`MjswanViewer.tsx` renders a Mantine **ControlPanel** + **Loader** and reaches into
engine internals. The only public contract today is `mount(el, source)` returning
`{ captureThumbnail, dispose }`. Consumers that want their own UI (mjswan Cloud's
playback overlay) cannot drive the simulation without the bundled React chrome.

Tracing the code shows the important fact: **the heavy engine is already
React-free.** `mjswanRuntime` ([runtime.ts](../../src/mjswan/template/src/core/engine/runtime.ts))
owns the WebGL context, the `setAnimationLoop` render loop, the `setTimeout` physics
loop, ONNX, splats, and the MuJoCo WASM module — no React. `CommandManager` and
`viewer_config` are plain TS too. React only owns four things, none of them the
engine:

1. **Selection orchestration** (`MountApp`): which project/scene/policy/motion/splat is
   current, and turning a change into a reload.
2. **Runtime lifecycle + MuJoCo load** (`MjswanViewer`): construct the runtime, load
   WASM, OOM-reload.
3. **Loading store** (`LoadingContext`).
4. **Chrome** (ControlPanel, Loader).

So the refactor is: **lift (1)–(3) out of React into a pure-TS engine, and demote
React to an optional client.** The goal is the best architecture, not compatibility.

## Decision

Delete `mount()`. Introduce a **pure-TS, instance-scoped engine** created by
`createEngine(element, options?)`. Three layers:

```
① mjswan (pure-TS engine)    createEngine / MjswanEngine / camera / commands.
                             No React, no catalog, no config.json, no fetch hijack.
                             One loaded simulation at a time.
② mjswan/manifest (plain TS) config.json + a byte source → a typed catalog.
                             Used by ③ and by Cloud. Framework-free.
③ React reference app        ControlPanel + Loader, a client of ①②. In-repo,
                             Python-built for standalone / demos / mjlab / Pages.
                             NOT published to npm (Cloud builds its own overlay).
```

The design decisions below were resolved one at a time; each cites the constraint it
serves.

### 1. Boundary — the app owns the catalog; the engine runs one simulation

MuJoCo holds exactly one live model, so the engine is inherently single-simulation.
"Multiple scenes/policies/splats/motions" is just a **catalog of loadable options**,
and switching has three distinct costs already encoded in the runtime:

- **scene** → `loadEnvironment` = full model rebuild.
- **policy** → `loadPolicyConfig` = swap ONNX/policy.json on the *same* model (no rebuild).
- **motion / splat** → `setSelectedMotion` / `setSplat` = live, no rebuild.

That cost knowledge is engine-domain. So rather than pushing a single opaque
`load(everything)` to the app (which would force every consumer to re-derive the
costs), **the app owns the catalog + current selection + next/prev, and the engine
exposes verbs that match the real costs**: `loadScene` (rebuild) vs `setPolicy` /
`setSplat` / `setMotion` (live). The app picks the verb; the engine guarantees the
cost.

Removed from the engine: `config.json` parsing, the `projects` layer (multi-project is
dropped — a sim is a single-project catalog), URL path resolution, and the `pick*`
helpers. `config.json` remains the Python-Builder↔consumer contract, but **parsing it
moves to `mjswan/manifest`** (layer ②) so the React app and Cloud share one parser
instead of each re-implementing it. `next`/`previous` cease to exist in the engine —
they are app one-liners over the catalog the app holds.

### 2. Instance-scoped, multiple instances per page, no globals

Today `getCommandManager()` and `SceneCacheManager.getInstance()` are process globals,
and `runtime.dispose()` carries a workaround comment: it must
`SceneCacheManager.resetInstance()` "*so the next mount rebuilds against its own module
(else restoreFromCache hits a cross-module embind mismatch)*." That is a global-state
bug already biting multi-mount.

**Everything becomes instance-scoped.** The `CommandManager` and any per-scene state are
owned by the `MjswanEngine` instance; there are no module globals. `dispose()` frees
exactly what that instance allocated (WASM module, GL context, loops, ONNX, splat) with
no singleton-reset black magic. **Multiple engines per page are supported** (replicate
galleries; a hidden thumbnail engine), bounded only by the browser's ~16 live WebGL
contexts and total WASM memory — a budget the app manages.

### 3. No engine-side scene cache

`SceneCacheManager` retained built scenes (`mjModel`/`mjData` + three.js graph) keyed by
path. Under the new model:

- The **byte layer** (don't re-download) is now the app's — it owns assets and can hold
  `ArrayBuffer`s and re-feed them.
- The **built layer** cannot cross the app/engine boundary as data: `mjModel`/`mjData`
  are embind-bound to *this* engine's WASM module and the three.js resources to *this*
  GL context — handing them out is exactly the cross-module bug above.
- Retaining multiple built scenes fights the non-shrinkable **2 GB WASM heap** (the
  reason OOM currently forces a page reload).

So the cache subsystem (`SceneCacheManager` / `resourceTracker` / `memoryMonitor`) is
**deleted**. The engine rebuilds on each `loadScene`. Warm caching, if needed, is an
**app** decision realized either as byte-retention (rebuild from memory, no download) or
as **multiple engine instances** (one per warm scene, zero rebuild, app-bounded
memory). Resource/memory risk becomes the app's explicit call.

Relatedly, **OOM ownership moves to the app**. Today `MjswanViewer` catches
`WasmMemoryLimitError` and calls `window.location.reload()`; the new engine must not own a
page reload (a host/platform action, like fullscreen). The engine surfaces the error —
`loadScene` rejects and `state.error` is set — and the host decides recovery (reload, drop
a warm instance, show a message).

### 4. Byte delivery — direct inputs; delete the `fetch` hijack

Every asset today flows through `fetch(baseUrl + path)`, with
[localAssets.ts](../../src/mjswan/template/src/core/utils/localAssets.ts)
**monkeypatching `window.fetch`** to serve local bytes from a synthetic origin. With the
app owning assets, that indirection is unnecessary.

The engine takes **bytes directly** in method arguments and never fetches:

```ts
type Bytes = ArrayBuffer | (() => Promise<ArrayBuffer>);   // in hand, or lazy loader
```

`localAssets.ts` (the global `window.fetch` wrapper, synthetic origins, token
ref-counting) is **deleted** — a significant removal of global side effects. Lazy
loaders (`() => Promise<ArrayBuffer>`) preserve today's on-demand motion loading; meshes
/ textures ride inside the `.mjz` (eager). `policy.json` is passed as a **parsed
object**, opaque to the app and interpreted by the engine (no double-fetch). `.mjz`
unpacking stays in the engine.

### 5. `createEngine` lifecycle

Two explicit phases: prepare the engine (WASM + GL), then load content. This lets the
app subscribe and configure the camera *before* the first scene loads.

```ts
const engine = await createEngine(element, { multithreaded: false });
const off = engine.subscribe(state => …);
await engine.loadScene({ model, policy, splat, viewer });
```

- `createEngine` is `async` and loads the MuJoCo module up front.
- **Multithreading is a runtime option** (`multithreaded`, default `false`), replacing
  the build-time `__MUJOCO_MT__` define. `true` dynamically imports `mujoco/mt` and
  requires COOP/COEP (the app's responsibility): Cloud stays single-threaded; the Python
  SPA / mjlab (service-worker headers) opt in.
- Init options stay minimal (`multithreaded` only). Renderer knobs
  (antialias/shadows/tone-mapping) remain hardcoded until a consumer needs them.

### 6. Two channels: promises for requests, one snapshot stream for engine state

- **Promises / return values** for request→result: `loadScene`/`setPolicy`/`setSplat`
  resolve or reject; `setMotion` returns an `accepted` boolean (rejection lets the app
  revert its selection).
- **`subscribe(state => …)`** for **engine-originated** state the app can't infer from
  its own calls — chiefly: load progress messages during an `await`; command values that
  change *without* an app call (policy load introduces terms; `reset` and **auto-
  termination in the physics loop** reset values); and playback flips the engine makes
  (auto-start after load). It pushes an **immutable snapshot** (not granular typed
  events), so a host renders wholesale via `useSyncExternalStore` and never has to demux
  an event taxonomy. The internal `CommandManager` keeps its granular events; the engine
  folds them into the snapshot at one place.

Camera state is deliberately **excluded** from the snapshot (it changes every frame with
tracking/user drag) — read it on demand via `camera.get()` (§7).

### 7. Camera API

A fourth camera driver alongside (1) initial `ViewerConfig`, (2) body tracking, (3)
`OrbitControls` user drag. Vocabulary is **spherical, in MuJoCo coordinates** (matching
`ViewerConfig`); raw three.js `position` is never exposed.

```ts
engine.camera.set({ lookat?, distance?, azimuth?, elevation?, fovy? }); // one-shot overwrite
engine.camera.get(): CameraView;                                         // current values
engine.camera.frame(): void;                                             // re-fit to scene bounds (new capability)
```

Precedence: `set()` overwrites the current camera state; **body tracking continues**
after it (change the angle, still follow the subject); **user drag is always live**
(controls are never disabled). Tracking on/off + target come only from `loadScene`'s
`viewer` — not mutable at runtime (no speculative surface).

### 8. Playback + commands verbs

- **`play()` / `pause()` / `reset()`** (not `start/stop` — `stop` reads as teardown;
  teardown is `dispose()`). `play/pause` is universal media vocabulary and matches the
  overlay use case: `pause()` halts physics while rendering continues (orbit a frozen
  frame). `reset()` reinitializes sim state without changing `play`/`pause`.
- **`reset()` is first-class**; the magic `_system:reset` pseudo-command is removed from
  `CommandManager`, so the command list no longer carries a fake button.
- **Commands**: read via the snapshot (`commands` descriptors + `commandValues`); write
  via a namespace mirroring `camera`: `engine.commands.set(id, value)` /
  `engine.commands.trigger(id)`.
- **`setReferenceVisible(visible)`** toggles the reference-motion ghost for tracking
  policies (existing capability); it is a live display toggle the app owns, so it is not in
  the snapshot.

### 9. Public API (`mjswan`)

```ts
export type Bytes = ArrayBuffer | (() => Promise<ArrayBuffer>);

export interface CreateEngineOptions {
  /** Load `mujoco/mt` (SharedArrayBuffer; requires COOP/COEP). Default false. */
  multithreaded?: boolean;
}
export function createEngine(
  element: HTMLElement,
  options?: CreateEngineOptions,
): Promise<MjswanEngine>;

// ── content inputs (app-supplied bytes) ───────────────────────────
export interface SceneInput {
  model: Bytes;                    // .mjz (engine unpacks)
  policy?: PolicyInput | null;
  splat?: SplatInput | null;
  viewer?: ViewerConfig;
  plugins?: EnginePlugins;         // scene-scoped custom terms (events); trusted only — §10
}
export interface PolicyInput {
  config: object;                  // parsed policy.json; opaque to app, engine interprets
  onnx: Bytes;
  motions?: MotionInput[];
  plugins?: EnginePlugins;         // policy-scoped custom terms (obs/termination/command); §10
}
export interface MotionInput { name: string; data: Bytes; default?: boolean; }
export interface SplatInput { data: Bytes; collider?: Bytes; transform?: SplatTransform; }

// ── camera ─────────────────────────────────────────────────────────
export interface CameraView {
  lookat: [number, number, number]; distance: number;
  azimuth: number; elevation: number; fovy: number;
}
export interface CameraControls {
  set(view: Partial<CameraView>): void;
  get(): CameraView;
  frame(): void;
}

// ── commands ────────────────────────────────────────────────────────
// Named CommandDescriptor (not CommandTerm) to avoid colliding with the internal
// `CommandTerm` in core/command/types.ts, which is the term *implementation* interface.
export interface CommandDescriptor {
  id: string; group: string;
  type: 'slider' | 'checkbox' | 'button';
  label: string; min?: number; max?: number; step?: number;
}
export interface CommandControls {
  set(id: string, value: number): void;
  trigger(id: string): void;
}

// ── engine state (subscribe snapshot) ───────────────────────────────
export interface MjswanEngineState {
  phase: 'running' | 'paused';
  loading: boolean;
  loadingMessage: string | null;
  error: Error | null;
  commands: ReadonlyArray<CommandDescriptor>;
  commandValues: Readonly<Record<string, number>>;
}

// ── the engine ──────────────────────────────────────────────────────
export interface MjswanEngine {
  // content — verbs match switch cost (§1)
  loadScene(input: SceneInput): Promise<void>;          // full model rebuild
  setPolicy(input: PolicyInput | null): Promise<void>;  // live, keeps model
  setSplat(input: SplatInput | null): Promise<void>;    // live
  setMotion(name: string | null): Promise<boolean>;     // live; returns accepted
  setReferenceVisible(visible: boolean): void;          // motion ghost toggle

  // playback
  play(): void; pause(): void; reset(): void;

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
```

`mjswan/manifest` (sketch): `parseManifest(configJson, byteSource) → Catalog`, where a
`Catalog` is `scenes[]` (each with lazy `model` bytes, `policies[]`, `splats[]`; policies
with `motions[]`). It produces exactly the `SceneInput`/`PolicyInput` shapes the engine
consumes. The app holds the `Catalog`, tracks selection, computes next/prev, and calls
engine verbs. Full types are deferred to implementation.

### 10. Custom MDP terms — runtime plugins, and the path to safe arbitrary terms

Custom MDP terms (issue #32) let authors define arbitrary Observation / Termination /
Event / Command logic in TypeScript for cases the engine's built-in blocks can't express.
Today `_build_client.py` **inlines that TS into the engine bundle at build time**, which
(a) forces an engine rebuild per project (Node/vite) and (b) bakes author code into the
bundle so a pinned/shared engine can't be reused.

The engine already has the registration points (`Observations` / `Terminations` /
`Events` registries; `CustomCommands`) — they are merely *populated at build time*.
**Change only who populates them and when: runtime registration.**

```ts
export interface EnginePlugins {
  observations?: Record<string, ObservationCtor>;
  terminations?: Record<string, TerminationCtor>;
  events?:       Record<string, EventCtor>;
  commands?:     Record<string, CommandCtor>;
}
// SceneInput.plugins / PolicyInput.plugins — author terms handed to a pinned engine at load
```

Plugins ride with the **content that declares the terms**, not with `createEngine`: scene-
scoped terms (events) on `SceneInput`, policy-scoped terms (observations/terminations/
commands) on `PolicyInput`, so a live `setPolicy` can bring its own new terms. A single
engine loads many scenes, so an engine-level plugin set would be the wrong granularity.

The author's TS is compiled by `Builder` (esbuild) into a **standalone ESM asset emitted
beside `config.json`** — the engine bundle is never rebuilt; only a tiny per-project
module is compiled (Node needed only for custom-JS builds; declarative builds need no
Node). The app loads that module and passes it as `plugins`. This fits the "app feeds the
engine its assets" model exactly: a custom-term module is just one more asset.

**Security — the danger is *effects*, not code.** The right line is not "code vs data"
but **effectful vs effect-free**. Split "dangerous" into three:

1. **Capabilities (side effects)** — I/O, network, DOM, exfiltration, origin authority.
   The only class that is genuinely unsafe to accept from untrusted authors.
2. **Resource abuse** — infinite loops, memory blow-up (DoS).
3. **Correctness** — NaNs, breaking the sim. Not a security threat.

Key facts:

- **Capability danger is eliminable *by construction*, not by monitoring.** In an
  object-capability model a computation can only cause an effect it holds a reference to.
  Run a term in an environment with no ambient authority — no `fetch`/`document`/`import`
  — and it *cannot name* an effect; this is structural, not a runtime "restriction." The
  purest instance is a **WASM module instantiated with an empty import object**: a
  malicious module that needs an import simply fails to instantiate. No analysis, no
  sandbox-as-monitor.
- **The infinitude of possible MDPs is *not* the blocker.** All pure computation is safe
  regardless of cardinality — safety is about the effect alphabet, not the size of the
  computation space. By **Rice's theorem** you cannot *decide* whether arbitrary TS is
  effect-free (nor whether it halts) — but safety needs no such decision: you either
  withhold the capabilities (ocap) or use a language that cannot express them (the
  declarative IR). Both sidestep undecidability by construction.
- **Resource abuse (2) is the one thing that *cannot* be decided away** (halting problem),
  so any Turing-complete escape hatch needs a runtime *budget* (fuel/epoch + memory cap) —
  a quantity bound, not a behavior monitor.
- **Under this refactor the effectful class nearly vanishes.** The classic exception
  (async `.npz` loading in the Mimic terms) disappears because the **app now feeds all
  bytes** (§4) — the term just reads a data slot. Raycasting reduces to "expose the
  heightfield as a read slot + a `Raycast` primitive." So what still needs an engine change
  is a new *pure primitive/slot*, essentially never a new *effect*.

**Confirmed policy.** The arbitrary-TS custom-MDP flow is a **permanent, trusted-only
capability** — it lets mjswan track the latest RL research and raises the engine's ceiling,
so it is *kept*, not deprecated. It is available in trusted contexts (`mjswan view`, local,
mjlab, the in-repo React app), which load and register author modules directly. **mjswan
Cloud rejects it.** The on-ramp to Cloud is: **submit a PR adding the needed primitive/slot
to the engine; once merged and released, upload against that supported engine version.**
Cloud stays **declarative-only** — but "declarative" is made expressive enough (option B
below) that the restriction is rarely felt.

**Future menu (recorded, not built now) — if Cloud should accept more without a PR:**

- **A — ocap execution of effect-free code.** Compile the author term to a **no-import
  WASM** module (AssemblyScript/Rust/…) run with an empty import object + a fuel/memory
  budget. Capability-free *by construction*; effectful terms fail to instantiate. This is
  the only route that *actually* achieves "arbitrary yet structurally safe," and would relax
  Cloud from "declarative-only" to "**effect-free arbitrary code**" — strictly more
  permissive, still provably safe. Cost: WASM toolchain in `Builder`, a numeric slot ABI,
  fuel integration, porting DX from TS.
- **C — unified lowering.** Authors always write mjlab-style TS; `Builder` *lowers* it to
  the pure IR / no-import WASM, and lowering **fails with a diagnostic** ("needs primitive
  X") when it uses an un-exposed effect. The "PR a primitive" on-ramp becomes a compiler
  error, and one authoring surface auto-classifies into Cloud-safe vs trusted-only.
- **B — expand the pure declarative IR** (ADR 0003): add pure control-flow / reduction ops
  (`Select`, bounded `Map`/`Reduce`) so *combinations* of exposed slots are effectively
  unbounded. Safe by language design; validation stays set-membership. This is the
  near-term investment that shrinks the trusted-only tail.

A **QuickJS-in-WASM interpreter** was considered as the sandbox and is **not recommended**:
it is a heavier trusted surface and slower per step (~50 Hz) than no-import WASM, whose
safety is the browser's own capability model rather than an interpreter mjswan ships.

### 11. Packaging and build flow

- **One npm package with subpath exports**: `mjswan` → `createEngine` (the engine);
  `mjswan/manifest` → the parser. No monorepo split. The React app is **not** an npm
  export — it is in-repo source that Python builds.
- **The CDN lib build's entry becomes `createEngine`** and contains **no React/Mantine**,
  shrinking the self-contained ESM Cloud loads. The WASM de-inline plugin and the
  "browser-self-contained, no bare imports" contract carry over; both single-thread and
  `mt` WASM variants are bundled (single eager, `mt` lazy).
- **Deleted**: `mount.tsx`, `MountApp.tsx`, `MjswanViewer.tsx` (its logic moves into the
  engine), `localAssets.ts`, `SceneCacheManager` + friends, the `__MUJOCO_MT__` define,
  the `_system:reset` pseudo-command, and config/project parsing from the engine.
- **React reference app** (`App.tsx`, `ControlPanel`, `Loader`): rewritten as a client of
  `createEngine` + `mjswan/manifest`. It stays default-on for standalone/mjlab use (the
  mjviser-parity panel is preserved), and is the reference implementation that proves the
  API is sufficient.
- **Python `Builder.build()` frontend build: default ON, skippable when an artifact
  exists.** Building the frontend is the default; if a prebuilt result is present, an
  explicit option skips it. Custom-JS builds still compile the small per-project term
  module (esbuild) but never rebuild the engine.

## Considered options

- **Additive `controls` on the existing React `mount()`** (the prior draft). Rejected now
  that backward compatibility is dropped: it leaves React in the engine's critical path,
  keeps the global singletons and the `fetch` hijack, and bloats the CDN payload with
  React for headless consumers. The split is the better architecture.
- **Engine owns a single-project catalog + selection + next/prev.** Rejected: it keeps
  config/selection logic in the engine that the app must own anyway (it owns the assets),
  and re-introduces the "engine knows the catalog" coupling. Verbs-matching-cost gives the
  app ownership while keeping switch-cost knowledge in the engine.
- **Single opaque `load(everything)` with engine-side change detection (declarative
  reconcile).** Elegant, but a reconciler is speculative complexity the app doesn't need —
  the app already knows whether it is switching scene vs policy, so it picks the verb.
- **Keep build-time custom-JS inlining.** Rejected: forces per-project engine rebuilds and
  precludes a pinned/shared engine; runtime plugins remove both.
- **QuickJS-in-WASM sandbox for untrusted custom code on Cloud.** Rejected (§10): a heavier
  trusted surface and slower per step than no-import WASM. If Cloud is ever to accept
  arbitrary terms, ocap no-import WASM (option A) is the structural answer; otherwise the
  PR-a-primitive on-ramp keeps Cloud declarative-only.

## Consequences

- The engine's public boundary is "bytes in → simulation + state snapshot + camera/command
  verbs out." No platform look, no catalog, no fetching. Fullscreen, next/prev, and the
  overlay all live in the app.
- `dispose()` is total per instance; the singleton-reset workaround disappears; multi-
  instance is a supported, first-class case.
- Cloud rewrites its integration: `createEngine` + `mjswan/manifest` + its own overlay,
  no bundled React. The engine CDN bundle shrinks (React/Mantine dropped).
- The standalone SPA / Python demos keep the React app; it is demoted from "the product"
  to "a reference client," which is what validates the API surface.
- Custom terms stop forcing engine rebuilds; the arbitrary-TS flow is kept as a trusted-only
  power feature, Cloud stays declarative-only, and the on-ramp to Cloud is a PR that adds the
  needed primitive/slot (with ocap no-import WASM recorded as the structural route if Cloud
  should ever accept effect-free arbitrary code without a PR).
- This is a large, breaking refactor touching the engine, the React app, the Python build
  flow, and Cloud. It is deliberately taken now, before the codebase grows, per the
  project owner's direction.
