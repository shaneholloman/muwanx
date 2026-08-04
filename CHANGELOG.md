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

- `PolicyHandle.add_velocity_command` / `add_command_velocity` — both removed
  with no alias. Pass `commands={"velocity": mjswan.velocity_command(...)}` to
  `add_policy()` instead.
