# CONTEXT.md

## What mjswan is

mjswan is a Python framework that packages browser-based MuJoCo simulations with real-time ONNX policy control into interactive static web apps. The Python side builds the bundle; the browser client is TypeScript/React/three.js running mujoco-wasm for physics and onnxruntime-web for policy inference. Published on PyPI and npm; demos hosted on GitHub Pages.


## Repository layout

```
src/mjswan/          Python package source
  builder.py           Builder — top-level entry point
  app.py               MjswanApp — launch / serve / publish built apps
  publish.py           Publish a built dist/ to mjswan Cloud (data-file upload protocol)
  project.py           ProjectConfig / ProjectHandle
  scene.py             SceneConfig / SceneHandle
  policy.py            PolicyConfig / PolicyHandle
  motion.py            MotionConfig / MotionHandle
  splat.py             SplatConfig / SplatHandle (Gaussian Splat)
  command.py           Command terms (Slider, Button, Checkbox, velocity_command, ui_command)
  viewer.py     ViewerConfig
  utils.py             ZIP-DEFLATE bundling, XML path rewriting, name2id slug helper
  wandb_io.py       W&B motion artifact downloads
  _cli.py              Typer-based `mjswan` CLI + legacy entry points
  _build_client.py     Frontend build orchestration (npm/vite)
  adapters/            mjlab soft-dependency adapter + compat helpers
  envs/mdp/            MDP building blocks (actions/ subpackage, events, observations, terminations)
  managers/            observation_manager, event_manager, action_manager, termination_manager
  template/            TypeScript frontend (Vite + React + three.js + mujoco-wasm)

examples/            Runnable examples
  demo/                main demo (deployed to GitHub Pages); includes gentle_humanoid/ and musclemimic.py
  mjlab/               mjlab-compatible examples (defaults, g1_spinkick, myosuite, unitree_rl)
  colab/               Google Colab notebook example
  tutorial/            hello_world, minimum_policy, newton_cradle quickstarts

tests/               pytest suite
docs/                zensical (MkDocs-based) site — published to Read the Docs
typings/             MuJoCo stub generator script
scripts/             Maintenance scripts (e.g., sync_contributors.py)
assets/              Demo GIF and banner SVG
```


## Python object model (fluent API)

```
Builder(base_path, gtm_id, mt, debug)
  ├── Builder.from_mjlab(task_id, run_path=..., play=...) → Builder  # classmethod factory
  ├── .add_project_mjlab(task_id, run_path=..., play=...) → ProjectHandle
  └── .add_project(name, id) → ProjectHandle
        ├── .add_scene_mjlab(task_id, play=...) → SceneHandle
        └── .add_scene(name, model|spec, metadata) → SceneHandle
              ├── .add_policy(name, policy, ...) → PolicyHandle
              │     └── .add_motion(...) / .add_motion_wandb(...) → MotionHandle
              ├── .add_policy_wandb(run_path, ...) → list[PolicyHandle]
              ├── .add_splat(name, source|url, ...) → SplatHandle
              └── .set_viewer(ViewerConfig)

builder.build(output_dir) → MjswanApp
MjswanApp.launch(host, port, open_browser)   # blocking; Colab-aware
```

`Builder.from_mjlab(task_id, run_path=...)` is the one-liner shortcut for the common "visualize a single mjlab task" pattern; it delegates to the instance method `Builder.add_project_mjlab`, which creates a project, adds an mjlab scene, and optionally attaches all `model_*.pt` checkpoints from one or more W&B runs (converted to ONNX via mjlab+torch). For finer control, build manually: `add_project` → `ProjectHandle.add_scene_mjlab` → `SceneHandle.add_policy_wandb(...)`.

Each `*Handle` wraps a `*Config` dataclass — the handle is the fluent API, the config is the serializable state.

The package's `__init__.py` is the canonical public API. Re-exports cover: `Builder` / `MjswanApp`; the five `*Handle` and `*Config` pairs; mjlab-compatible MDP cfgs (`ObservationGroupCfg`, `ObservationTermCfg`, `ActionTermCfg`, `JointPositionActionCfg`, `JointEffortActionCfg`, `TerminationTermCfg`); command UI (`SliderConfig`/`ButtonConfig`/`CheckboxConfig` and their `Slider`/`Button`/`Checkbox` aliases, `CommandTermConfig`, `ui_command`, `velocity_command`); and the `register_observation` / `register_event` / `register_termination` / `register_command` extension hooks.


## Key modules

### `builder.py` — `Builder`
Main entry point. Accumulates `ProjectConfig` objects and calls `ClientBuilder` to invoke the Vite frontend build, then writes `config.json` + per-scene DEFLATE-compressed ZIPs (via `utils.to_zip_deflated`, since `mujoco.to_zip` stores entries uncompressed) plus policy/motion/splat assets into the output directory.

### `app.py` — `MjswanApp`
Wraps a built `dist/` directory. `launch()` starts a stdlib HTTP server (COOP/COEP headers required for SharedArrayBuffer / MuJoCo WASM threading); detects Google Colab and displays an inline iframe instead.

### `policy.py` — `PolicyConfig` / `PolicyHandle`
Holds an `onnx.ModelProto` plus observation groups, action terms, termination terms, commands, and motion references. Compatible with mjlab config classes via the adapter layer. Serialized to a per-policy `<name>.json` at build time.

### `command.py`
Defines command terms consumed by policies: `SliderConfig`, `ButtonConfig`, `CheckboxConfig` (aliased as `Slider` / `Button` / `Checkbox`), `CommandTermConfig`, `CommandBinding`, `CommandUiConfig`, and the `CommandInput` union of input types. `velocity_command()` is a convenience factory for the standard locomotion 3-DoF velocity command, and `ui_command()` builds a generic UI-driven command term. Custom command terms can be registered with `register_command`.

### `scene.py` — `SceneConfig` / `SceneHandle`
A scene owns one MuJoCo model (as `MjModel` → binary `.mjb` or `MjSpec` → XML), zero or more policies, and zero or more Gaussian splat backgrounds.

### `splat.py` — `SplatConfig` / `SplatHandle`
Configures a 3D Gaussian Splat (`.spz` format) background: scale, position offsets, Euler rotations, optional collider mesh URL.

### `viewer.py` — `ViewerConfig`
Camera parameters (lookat, distance, fovy, elevation, azimuth) + tracking mode (`OriginType`: AUTO / WORLD / ASSET_ROOT / ASSET_BODY). `ViewerConfig.from_position()` computes spherical params from a Cartesian viewer position.

### `adapters/`
- `mjlab_adapter.py`: Converts mjlab types (observations, actions, terminations, events, commands) to mjswan equivalents by name-based dynamic lookup — no hardcoded registries, no hard import of mjlab.
- `mjlab_compat.py`: Monkey-patches `MujocoCfg.apply_to_spec()` onto mjlab when needed.

### `envs/mdp/` and `managers/`
mjlab-compatible MDP layer. `envs/mdp/` holds the config-side building blocks: observation groups, action terms (`JointPositionActionCfg`, `JointEffortActionCfg`, `MuscleActivationActionCfg`), event functions, and termination functions. `managers/` holds the runtime counterparts (`observation_manager`, `event_manager`, `action_manager`, `termination_manager`) that mjlab's training loop hands to the policy. Custom obs/event/termination functions are registered via `register_observation` / `register_event` / `register_termination`.

**Muscle action term.** `MuscleActivationActionCfg` drives MuJoCo muscle actuators. `normalize=True` (default) applies the canonical MyoSuite sigmoid `σ(5(scale·a + offset − 0.5))` to map policy outputs into excitation in (0, 1); `normalize=False` clips `scale·a + offset` to [0, 1]. The semantics mirror myosuite4's `MuscleActionTermCfg.normalize`. The mjlab adapter translates `MyoMuscleActivationActionCfg` (the class actually used by every myo* mjlab task) to `MuscleActivationActionCfg`; see [docs/adr/0002](./docs/adr/0002-muscle-action-term-aligned-with-myomuscleactivationactioncfg.md).

### `_build_client.py`
Orchestrates the Node.js / Vite frontend build. Manages a local `nodeenv` if Node isn't available system-wide.

### `wandb_io.py`
Downloads motion `.npz` artifacts from Weights & Biases runs. Used by `PolicyHandle.add_motion_wandb()`.

### `utils.py`
Asset bundling and path helpers. `to_zip_deflated()` is the per-scene packager: it collects mesh/texture/hfield/skin files from disk (with `spec.assets` fallback for mjlab's prefixed-key layout), encodes buffer-only textures as PNGs, rewrites the MuJoCo XML so meshdir/texturedir hints are eliminated and all paths are ZIP-safe, and writes a DEFLATE-compressed ZIP that JSZip decodes on the client. `name2id()` is the lowercase-underscore slug helper used everywhere project / scene / policy IDs are derived from human-readable names.


## Frontend (`src/mjswan/template/`)

TypeScript + React + Vite + three.js. Built by `Builder.build()` via `_build_client.py`. The browser client:
- Loads the MuJoCo WASM module and runs physics in a Web Worker.
- Runs ONNX policies via onnxruntime-web.
- Renders via three.js (reflections, shadows, Gaussian Splat background).
- Supports WebXR (VR).
- Reads `config.json` to discover projects/scenes/policies at runtime.

Multi-threaded mode (`Builder(mt=True)`) requires COOP/COEP headers; the builder writes a `_headers` file (Netlify / Cloudflare Pages / Vercel) and a service-worker script (required for GitHub Pages).

The template has two Vite build outputs (both written to `template/dist/`):
- **SPA** (`vite.config.ts`, `npm run build:spa`) — the standalone app the Python `Builder` assembles. Entry `src/index.tsx`.
- **Library** (`vite.lib.config.ts`, `npm run build:lib`) — a single self-contained ESM `dist/mjswan.js` exposing `mount(element, configUrl)` (`src/mount.tsx` → `MountApp.tsx`), consumed by mjswan Cloud from a CDN. Every dependency is bundled (no bare imports) and the MuJoCo/ONNX WASM is emitted as co-located files referenced via `new URL('./x.wasm', import.meta.url)`. Vite lib mode force-inlines those WASM as base64; a `generateBundle` plugin in `vite.lib.config.ts` extracts them back into co-located files. `npm run build` runs both. Shared config-shape + selection helpers live in `src/core/appConfig.ts` (used by both entries). See mjswan-cloud ADR 0001.


## CLI entry points

The primary CLI is `mjswan` (Typer-based, defined in `_cli.py:app`). Subcommands:

| Subcommand | Description |
|------------|-------------|
| `mjswan view <model.xml>` | Build and launch a viewer for a MuJoCo XML/MJCF file |
| `mjswan serve <dist-dir>` | Serve a pre-built `dist/` directory |
| `mjswan new <name> [--template hello-world\|policy\|mjlab]` | Scaffold a new project from a template |
| `mjswan demo [name]` / `--list` | Run a built-in demo (`simple`, `main`, `mjlab`) |
| `mjswan info <dist-dir>` | Show a tree of projects/scenes/policies and asset sizes |
| `mjswan publish <dist-dir>` | Upload a built dist's data files to mjswan Cloud (rejects custom-JS builds) |

Legacy entry points (kept for backward compatibility): `main`, `simple`, `mjlab`, `serve <dist-dir>` — each runs the corresponding `examples/` module.


## Tooling and workflow

| Tool | Purpose |
|------|---------|
| `uv` | Dependency management and script runner — use instead of bare `python`/`pip` |
| `hatchling` | Build backend |
| `ruff` | Linting and formatting |
| `pyright` / `ty` | Type checking |
| `pytest` | Tests (`make test`) |
| `pre-commit` | Hooks: trailing-whitespace, ruff, pytest (not slow), eslint |
| `zensical` | Docs site builder (MkDocs-based) — `make docs-build` / `make docs-serve` |

Key Makefile targets: `sync`, `format`, `type`, `check`, `test`, `test-all`, `docs-build`, `docs-serve`.


## Test markers

`@pytest.mark.slow` triggers a full frontend (npm + Vite) build and is excluded from pre-commit (`pytest -m "not slow"`); unmarked tests are fast and always run. CI (`pytest.yml`) runs `pytest -m "not slow"` across Python 3.10 / 3.11 / 3.12.


## Dependencies

Core: `mujoco==3.8.1`, `onnx>=1.20.0`, `nodeenv>=1.9.1`, `rich>=13.0.0`, `wandb>=0.23.1`, `typer` (for the `mjswan` CLI).
Dev extras: `pyright`, `ruff`, `pre-commit`, `pytest`.
Examples extras: `mjlab`, `torch`, `robot-descriptions`, `playground`, `myosuite`, `gymnasium`.

mjlab itself pulls in `mujoco-mjx==3.8.1` and `mujoco-warp>=3.8.0.3` (3.8.0.3 switched from `mjENBL_MULTICCD` to a `DisableBit`, restoring compat with stable mujoco 3.8.1).

Python 3.10–3.12 only.


## Deployment

The demo app is built by `examples/demo/main.py` and deployed to GitHub Pages via the `deploy.yml` workflow on every push to `main` that touches relevant paths. The `MJSWAN_BASE_PATH` and `MJSWAN_NO_LAUNCH` env vars control the build.
