# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

API-wide rename for naming consistency. Methods that add an object now follow
`add_[layer]_[source]`; toggles use `enable_`/`set_`; the MDP binding layer is
spelled out. **All pre-0.8 names remain importable as aliases via
`mjswan/_compat.py` until 0.9; renamed methods, modules, and `register_*`
functions emit a `DeprecationWarning`, the MDP binding *class* aliases stay
silent (a type alias cannot warn on attribute access). The velocity-command
shortcuts were removed outright (no alias) — see Removed.**

### Added

- **MDP term bodies are traced to ONNX at build time and run by ONNX Runtime Web**
  ([ADR 0005](docs/adr/0005-onnx-traced-terms-superseding-the-declarative-dsl.md)),
  replacing the hand-written TypeScript DSL — see Removed. mjlab's real
  observation / termination / event / command functions are exported as graphs, so
  there is no second copy of the math to keep in numeric lockstep and no closed
  primitive set to extend. A term with no browser implementation now fails the
  build instead of being dropped.
- Traced-graph coverage for mjlab's structured sensors: `RayCastSensor` (rays cast
  in the browser, completing `height_scan`) and `ContactSensor`.
- Seeded PRNG behind every term's randomness (`createEngine({ termSeed })`,
  reported back as `MjswanEngineState.termSeed`), so a recorded session replays.
- Debug visualisation for command terms, mirroring mjlab's `debug_vis` — arrows and
  markers are emitted as data by `default_viz()`, toggled via `engine.debugVis.set`,
  and on by default as in mjlab's viewer.
- `Builder.add_project_mjlab(task_id, ...)` — instance-method counterpart to the
  `Builder.from_mjlab` classmethod factory, for adding an mjlab task to a builder
  that already has other projects. `from_mjlab` now delegates to it.
- `ObservationTermCfg.history_steps` — sparse look-back offsets for a term, e.g.
  `(0, 1, 2, 4, 8, 16)`, where mjlab's `history_length` can only count frames.
  The runtime now stacks per-term history at all (previously the build emitted
  `history_length` and nothing read it, so per-term history was dropped).
- Look-ahead reference slots on the built-in `TrackingCommand`: `ref_root_pos_w`,
  `ref_root_quat_w`, `ref_joint_pos` (each the reference trajectory sampled at the
  command's `time_steps` offsets) and `is_ready`, for policies trained on a window
  of the clip rather than the current frame alone.
- `build_single_entity_trace_env(commands=...)` and `TraceCommandManager`, so a
  traced term can read a command that exists browser-side only.

### Changed

- **Methods**
  - `ProjectHandle.add_mjlab_scene` → `ProjectHandle.add_scene_mjlab`
  - `SceneHandle.add_policy_from_wandb` → `SceneHandle.add_policy_wandb`
  - `SceneHandle.set_viewer_config` → `SceneHandle.set_viewer`
  - `SceneHandle.add_splat_section` → `SceneHandle.enable_splat_section`
  - `PolicyHandle.add_motion_from_wandb` → `PolicyHandle.add_motion_wandb`
    (parameter `wandb_run_path` → `run_path`)
- **Classes**
  - `mjswanApp` → `MjswanApp` (deprecated alias kept until 0.9)
  - `ObsBinding` / `ObsFunc` → `ObservationBinding`
  - `TermBinding` / `TermFunc` → `TerminationBinding`
  - `EventFunc` → `EventBinding`
  - `CommandTermSpec` → `CommandBinding`
  - `MjlabMdpBinding` → `MdpBinding`
- **Functions**
  - `register_obs_func` → `register_observation`
  - `register_termination_func` → `register_termination`
  - `register_event_func` → `register_event`
  - `register_command_term` → `register_command`
- **Modules**
  - `mjswan.viewer_config` → `mjswan.viewer`
  - `mjswan.wandb_utils` → `mjswan.wandb_io`
- The built `dist/` no longer copies the unused `logo-color.svg` (only `logo.svg`).
- The `examples` extra pins `mjlab==1.5.3` exactly (was `>=1.3.0`) and moves to
  `mujoco` 3.10, adding `onnxruntime`. The pin is exact because the tracer reads
  mjlab's internals; a weekly CI parity sweep is what catches upstream drift.

### Deprecated

All kept as aliases via `_compat.py`, removed in 0.9:

- Renamed methods, modules, and `register_*` functions — emit a
  `DeprecationWarning`.
- The pre-0.8 MDP binding **class aliases** — `ObsBinding`, `ObsFunc`,
  `TermBinding`, `TermFunc`, `EventFunc`, `MjlabMdpBinding`, `CommandTermSpec` —
  restored as silent aliases on their original import paths. Migrate to the
  spelled-out `*Binding` names (`ObservationBinding`, `TerminationBinding`,
  `EventBinding`, `MdpBinding`, `CommandBinding`).

### Removed

- **`mjswan.dsl`** — the declarative composition-graph DSL (ADR 0003) and its
  TypeScript interpreter, removed outright with no alias. Term bodies are traced
  to ONNX instead (see Added), so `div` / `sqrt` / `slice_` / `normalize` /
  `quat_to_rot6d_columns` and the rest have no successor: write the term as an
  ordinary mjlab-style Python function against the live env and let the build
  trace it. `scripts/verify_dsl_migration.py` goes with it.
- `PolicyHandle.add_velocity_command` / `add_command_velocity` — both removed
  with no alias. Pass `commands={"velocity": mjswan.velocity_command(...)}` to
  `add_policy()` instead.
