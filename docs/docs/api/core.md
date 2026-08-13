---
icon: octicons/package-16
---

# Core API

This page documents every public symbol exported from the `mjswan` package.

---

## Builder

```python
class mjswan.Builder(
    base_path: str = "/",
    gtm_id: str | None = None,
    mt: bool = False,
    debug: bool = False,
)
```

Top-level builder that orchestrates projects, scenes, policies, and splats and produces a deployable web application.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `base_path` | `str` | `"/"` | URL prefix for subdirectory deployments. Set to e.g. `"/mjswan/"` when the site lives at `https://user.github.io/mjswan/`. |
| `gtm_id` | `str \| None` | `None` | Google Tag Manager container ID (e.g. `"GTM-XXXXXXX"`). When provided, the GTM snippet is injected into the built HTML. |
| `mt` | `bool` | `False` | Enable multi-threaded MuJoCo WASM. Requires Cross-Origin Isolation; mjswan emits a `_headers` file (Netlify / Cloudflare Pages / Vercel) and a `coi-serviceworker.js` (required for GitHub Pages, which cannot set response headers). |
| `debug` | `bool` | `False` | Keep browser console messages in the built application. Defaults to stripping them from the production bundle. |

### Builder.from_mjlab

```python
@classmethod
def from_mjlab(
    task_id: str,
    *,
    run_path: str | list[str] | None = None,
    project_name: str = "mjlab",
    play: bool | None = None,
    env_cfg: Any | None = None,
    base_path: str = "/",
    gtm_id: str | None = None,
    mt: bool = False,
    debug: bool = False,
) -> Builder
```

Convenience factory that creates a `Builder` pre-configured with a single mjlab task. Delegates to the instance method `Builder.add_project_mjlab`. The returned `Builder` already contains one project and one scene; call `build()` directly, or modify it further before building.

`play` and `env_cfg` behave exactly as on `ProjectHandle.add_scene_mjlab`, including being mutually exclusive; both are forwarded unresolved.

When `run_path` is supplied, every `model_*.pt` checkpoint from each W&B run is fetched and converted to ONNX via mjlab + torch (both required). Each attached policy configures itself from the task — observations, commands, actions and terminations from its `env_cfg`, `clip_actions` from its runner config. For finer control, build manually with `add_project` → `ProjectHandle.add_scene_mjlab` → `SceneHandle.add_policy_wandb`.

**Returns** — `Builder`

### Builder.add_project_mjlab

```python
def add_project_mjlab(
    task_id: str,
    *,
    run_path: str | list[str] | None = None,
    project_name: str = "mjlab",
    play: bool | None = None,
    env_cfg: Any | None = None,
) -> ProjectHandle
```

Add a project pre-configured with a single mjlab task (project + mjlab scene + optional W&B policies). The instance-method counterpart to `Builder.from_mjlab`; use it to add an mjlab task to a builder that already has other projects.

**Returns** — `ProjectHandle`

### Builder.add_project

```python
def add_project(name: str, *, id: str | None = None) -> ProjectHandle
```

Add a project to the application.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the UI. |
| `id` | `str \| None` | `None` | URL slug. The first project defaults to `None` (served at `/`). Subsequent projects without an explicit `id` get one derived from `name` (lowercased, spaces/hyphens → underscores). |

**Returns** — `ProjectHandle`

### Builder.build

```python
def build(output_dir: str | Path | None = None) -> MjswanApp
```

Compile and save the application.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str \| Path \| None` | `None` | Output directory. Defaults to `dist/` next to the calling script. Relative paths are resolved against the caller's directory. |

**Returns** — `MjswanApp`

**Raises** — `ValueError` if no projects have been added.

### Builder.get_projects

```python
def get_projects() -> list[ProjectConfig]
```

Return a copy of all project configurations.

---

## ProjectHandle

Returned by `Builder.add_project()`. Use it to add scenes to a project.

### ProjectHandle.add_scene

```python
def add_scene(
    name: str,
    *,
    model: mujoco.MjModel | None = None,
    spec: mujoco.MjSpec | None = None,
    metadata: dict[str, Any] | None = None,
    control_dt: float | None = None,
    events: Mapping[str, Any] | None = None,
) -> SceneHandle
```

Add a MuJoCo scene. Provide exactly one of `model` or `spec`.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the UI. |
| `model` | `mujoco.MjModel \| None` | `None` | Compiled MuJoCo model. Saved as `.mjb` (binary). Loads faster; larger files. |
| `spec` | `mujoco.MjSpec \| None` | `None` | MuJoCo spec. Saved as `.mjz` (DEFLATE-compressed ZIP). Smaller files; slightly slower to load. |
| `metadata` | `dict \| None` | `None` | Arbitrary key-value metadata stored in `config.json`. |
| `control_dt` | `float \| None` | `None` | Seconds per control step — mjlab's `timestep * decimation`. Required once the scene carries a policy: the model holds only the physics timestep, and a wrong control rate raises nothing at playback. `add_scene_mjlab` fills it in from the task. |
| `events` | `Mapping[str, Any] \| None` | `None` | Scene events (`EventTermCfg` instances, mjswan or mjlab). Same as calling `SceneHandle.set_events` afterwards. Events are scene-scoped, not per-policy: the runtime keeps one `EventManager` per scene across policy switches, and `mode="startup"` fires once at scene load, before any policy is chosen. |

**Returns** — `SceneHandle`

**Raises** — `ValueError` if both or neither of `model`/`spec` are provided.

### ProjectHandle.add_scene_mjlab

```python
def add_scene_mjlab(
    task_id: str,
    *,
    play: bool | None = None,
    env_cfg: Any | None = None,
    events: Mapping[str, Any] | None = None,
) -> SceneHandle
```

Load an mjlab task's MuJoCo spec from the task registry and add it as a scene. Requires `mjlab` to be installed. Automatically applies the task's `viewer`, `events`, and any terrain data — including swapping mjlab's `reset_root_state_uniform` for a spawn on a random flat terrain patch, since the browser runs a single env where mjlab trains many spread across the terrain.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str` | — | mjlab task identifier (e.g. `"go2_flat"`). |
| `play` | `bool \| None` | `None` | Which of the task's two registered configs to load. mjlab keeps them as `env_cfg` (training) and `play_env_cfg`; this selects between them exactly as its `load_env_cfg(task_id, play=...)` does. **Unset means play — the opposite of mjlab's own default, deliberately**: that one serves training scripts, and this is a playback tool. mjlab's training config sets `episode_length_s` to 10–20 s, which mjswan serializes into the browser's `time_out` termination, so a viewer built from it resets the robot every few seconds; it also keeps `push_robot` and the terrain-bounds termination, and lacks `randomize_terrain`. Pass `False` to reproduce training-time conditions. **Mutually exclusive with `env_cfg`** — passing both raises. |
| `env_cfg` | `Any \| None` | `None` | Pre-loaded (and possibly edited) env config to use instead of loading `task_id` fresh. Load it with the `play` you want — `load_env_cfg(task_id, play=True)` — since `play` here then has nothing left to select. The scene keeps whichever config it used, and policies added to it default their term sets to it. A tracking task does not need this: mjlab registers it with `commands["motion"].motion_file = ""`, and the builder points that at the clip it bundles. |
| `events` | `Mapping[str, Any] \| None` | `None` | Scene events, overriding the task's own `env_cfg.events`. Omit to take the task's; pass `{}` for a scene with none. |

**Returns** — `SceneHandle`

**Raises** — `ImportError` if `mjlab` is not installed.

### ProjectHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the project. |
| `id` | `str \| None` | URL slug of the project. |

---

## SceneHandle

Returned by `ProjectHandle.add_scene()`. Use it to attach policies, splats, viewer config, and reset events to a scene.

### SceneHandle.add_policy

```python
def add_policy(
    name: str,
    policy: onnx.ModelProto,
    *,
    metadata: dict[str, Any] | None = None,
    config_path: str | None = None,
    source_path: str | None = None,
    env_cfg: Any | None = None,
    task_id: str | None = None,
    observations: ObservationGroupCfg | Mapping[str, Any] | None = None,
    commands: Mapping[str, CommandTermConfig] | None = None,
    actions: Mapping[str, ActionTermCfg] | None = None,
    terminations: dict[str, TerminationTermCfg] | None = None,
    policy_joint_names: list[str] | None = None,
    policy_num_actions: int | None = None,
    default_joint_pos: list[float] | None = None,
    encoder_bias: list[float] | None = None,
    clip_actions: float | None = None,
    initial_qpos: list[float] | None = None,
    initial_qvel: list[float] | None = None,
    extras: dict[str, Any] | None = None,
    default: bool = False,
) -> PolicyHandle
```

Attach an ONNX policy to the scene. `observations`, `commands`, `actions`, and `terminations` all accept mjlab-compatible config classes (mjswan converts them via the adapter layer; mjlab is a soft dependency), and each defaults to the matching field of the scene's mjlab env config when it has one — pass `{}` for a policy that genuinely has none.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the UI. |
| `policy` | `onnx.ModelProto` | — | Loaded ONNX model (e.g. from `onnx.load("policy.onnx")`). |
| `metadata` | `dict \| None` | `None` | Arbitrary key-value metadata. |
| `source_path` | `str \| None` | `None` | Path to the source `.onnx` file. Written to `config.json` for reference. |
| `config_path` | `str \| None` | `None` | Path to a JSON file describing observations / actions / etc. mjswan merges any Python-side `commands`/`observations`/`actions`/`terminations` into this file. See [Policy Config Format](../notes/policy-config.md). |
| `env_cfg` | `Any \| None` | `None` | mjlab env config to take this policy's unset term sets from, instead of the scene's. Its control rate must match the scene's `control_dt`. |
| `task_id` | `str \| None` | `None` | mjlab task id used to read the task's runner config (which observation group the actor reads, and `clip_actions`). Defaults to the scene's task. |
| `observations` | `ObservationGroupCfg \| dict[str, ObservationGroupCfg] \| None` | `None` | A single observation group — mjlab's `env_cfg.observations["actor"]` — or a dict of them keyed by **ONNX input tensor name**. Prefer the single group: the key is an input name the runtime feeds, not a label, and a wrong one fails silently at playback. A `"critic"` group is dropped with a warning (only the actor is exported to ONNX). Accepts both mjswan and mjlab `ObservationGroupCfg` instances. |
| `commands` | `Mapping[str, CommandTermConfig] \| None` | `None` | Command terms keyed by policy-visible name (e.g. `"velocity"`). Use `mjswan.velocity_command()` or `mjswan.ui_command([...])` to construct values. Accepts mjlab `CommandTermCfg` instances too. |
| `actions` | `Mapping[str, ActionTermCfg] \| None` | `None` | Action term configs keyed by name (e.g. `"joint_pos"`). |
| `terminations` | `dict[str, TerminationTermCfg] \| None` | `None` | Termination term configs keyed by name. |
| `policy_joint_names` | `list[str] \| None` | `None` | Ordered list of joint names the policy controls. Required by the browser runtime to map outputs to actuators. |
| `policy_num_actions` | `int \| None` | `None` | Output width for policies whose action count cannot be inferred from `policy_joint_names` — e.g. muscle-driven ones, which drive actuators rather than joints. |
| `default_joint_pos` | `list[float] \| None` | `None` | Default (resting) joint positions corresponding to `policy_joint_names`. |
| `encoder_bias` | `list[float] \| None` | `None` | Per-joint encoder bias (mirrors mjlab's joint-position action path). |
| `clip_actions` | `float \| None` | `None` | Symmetric bound on the raw policy output, applied before any action term sees it (rsl-rl's `RslRlVecEnvWrapper`). Distinct from `ActionTermCfg.clip`, which bounds `raw * scale + offset` per target. Defaults to the task's runner config; `0.0` is a real bound. |
| `initial_qpos` | `list[float] \| None` | `None` | Optional initial qpos serialized into the policy JSON for reset logic. |
| `initial_qvel` | `list[float] \| None` | `None` | Optional initial qvel serialized into the policy JSON for reset logic. |
| `extras` | `dict \| None` | `None` | Extra JSON payload merged verbatim into the generated policy config. |
| `default` | `bool` | `False` | If `True`, this policy is initially selected in the viewer. |

**Returns** — `PolicyHandle`

### SceneHandle.add_policy_wandb

```python
def add_policy_wandb(
    run_path: str | list[str],
    *,
    only_latest: bool = False,
    task_id: str | None = None,
    config_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    env_cfg: Any | None = None,
    observations: ObservationGroupCfg | dict[str, ObservationGroupCfg] | None = None,
    commands: Mapping[str, Any] | None = None,
    actions: Mapping[str, ActionTermCfg] | None = None,
    terminations: dict[str, TerminationTermCfg] | None = None,
    clip_actions: float | None = None,
    extras: dict[str, Any] | None = None,
) -> list[PolicyHandle]
```

Fetch ONNX policies from one or more W&B runs and attach them all to the scene. Same `observations` / `commands` / `actions` / `terminations` are applied to every policy.

When `only_latest=False` (the default), all `model_*.pt` checkpoints in each run are downloaded and converted to ONNX via mjlab + torch — `task_id` is required, and comes from the scene unless the scene is a plain one. When `only_latest=True`, only the exported `.onnx` artifact is fetched.

Every term set defaults to the scene's mjlab env config (or to `env_cfg=`, when given), and `task_id` to the scene's task — so for a scene from `add_scene_mjlab` the run path alone is enough. `clip_actions` is read from the task's runner config automatically.

**Returns** — `list[PolicyHandle]` (flat across all runs). The latest checkpoint (highest `_<step>` suffix) is marked as the default.

**Raises** — `ValueError` if `only_latest=False` and `task_id` is missing; `ImportError` if `mjlab`/`torch` are missing.

### SceneHandle.add_splat

```python
def add_splat(
    name: str,
    *,
    source: str | None = None,
    url: str | None = None,
    scale: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    collider_url: str | None = None,
    control: bool = False,
) -> SplatHandle
```

Add a Gaussian Splat background to the scene. Exactly one of `source` or `url` must be supplied.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the viewer selector. |
| `source` | `str \| None` | `None` | Local path to a `.spz` file. The file is copied into `dist/` during `Builder.build()`. Mutually exclusive with `url`. |
| `url` | `str \| None` | `None` | URL to an external `.spz` file. Fetched by the browser at runtime; not bundled. Mutually exclusive with `source`. |
| `scale` | `float` | `1.0` | Metric scale factor (converts splat units to metres). |
| `x_offset` | `float` | `0.0` | X-axis position offset in scaled splat units. |
| `y_offset` | `float` | `0.0` | Y-axis position offset in scaled splat units. |
| `z_offset` | `float` | `0.0` | Vertical position offset. Use `ground_plane_offset` from capture metadata if available. |
| `roll` | `float` | `0.0` | Roll rotation in degrees applied on top of the COLMAP → Three.js base rotation. |
| `pitch` | `float` | `0.0` | Pitch rotation in degrees applied on top of the COLMAP → Three.js base rotation. |
| `yaw` | `float` | `0.0` | Yaw rotation in degrees applied on top of the COLMAP → Three.js base rotation. |
| `collider_url` | `str \| None` | `None` | Optional URL or local path to a `.glb` collision mesh. |
| `control` | `bool` | `False` | Show live scale/offset/rotation controls in the viewer control panel. Useful during calibration. |

**Returns** — `SplatHandle`

**Raises** — `ValueError` if both or neither of `source`/`url` are provided.

### SceneHandle.enable_splat_section

```python
def enable_splat_section() -> SceneHandle
```

Show the Splat selector in the control panel even when no splats are pre-configured. This lets users load a `.spz` file by pasting an external URL at runtime, without requiring any `add_splat()` calls.

Has no effect when at least one splat is already attached (the selector is shown automatically in that case).

Returns `self` for chaining.

### SceneHandle.set_viewer

```python
def set_viewer(config: ViewerConfig) -> SceneHandle
```

Set the camera, tracking mode, and rendering options for the scene. See [`ViewerConfig`](#viewerconfig).

Returns `self` for chaining.

### SceneHandle.set_events

```python
def set_events(events: Mapping[str, Any]) -> SceneHandle
```

Set scene-level reset events. Accepts a dict of `EventTermCfg` instances (mjswan or mjlab). Only events with `mode="reset"` are forwarded to the browser runtime.

Returns `self` for chaining.

### SceneHandle.set_metadata

```python
def set_metadata(key: str, value: Any) -> SceneHandle
```

Set a metadata entry for the scene. Returns `self` for chaining.

### SceneHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the scene. |

---

## PolicyHandle

Returned by `SceneHandle.add_policy()`. Use it to attach commands, motions, and metadata.

Note that command groups are normally passed to `add_policy(commands=...)` directly. For the standard locomotion case pass `commands={"velocity": mjswan.velocity_command(...)}`.

### PolicyHandle.add_motion

```python
def add_motion(
    *,
    name: str,
    source: str,
    fps: float = 50.0,
    anchor_body_name: str,
    body_names: tuple[str, ...] | list[str],
    dataset_joint_names: list[str] | None = None,
    default: bool = False,
    loop: bool = True,
) -> MotionHandle
```

Attach a bundled `.npz` reference motion to the policy (used by motion-tracking policies).

**Returns** — `MotionHandle`

### PolicyHandle.add_motion_wandb

```python
def add_motion_wandb(
    *,
    name: str | None = None,
    run_path: str | None = None,
    run_id: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    fps: float = 50.0,
    anchor_body_name: str,
    body_names: tuple[str, ...] | list[str],
    dataset_joint_names: list[str] | None = None,
    default: bool = False,
    loop: bool = True,
) -> MotionHandle
```

Download a motion `.npz` artifact from a W&B run and attach it to the policy. Supply either `run_path="entity/project/run_id"` or the three pieces separately.

**Returns** — `MotionHandle`

### PolicyHandle.set_metadata

```python
def set_metadata(key: str, value: Any) -> PolicyHandle
```

Set a metadata entry for the policy. Returns `self` for chaining.

### PolicyHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the policy. |
| `model` | `onnx.ModelProto` | The attached ONNX model. |

---

## SplatHandle

Returned by `SceneHandle.add_splat()`.

### SplatHandle.set_metadata

```python
def set_metadata(key: str, value: Any) -> SplatHandle
```

Set a metadata entry for the splat. Returns `self` for chaining.

### SplatHandle properties

| Property | Type | Description |
|---|---|---|
| `source` | `str \| None` | Local path to the bundled `.spz` file, or `None` if `url` was used. |
| `url` | `str \| None` | External URL to the `.spz` file, or `None` if `source` was used. |
| `scale` | `float` | Metric scale factor. |
| `x_offset` | `float` | X-axis position offset. |
| `y_offset` | `float` | Y-axis position offset. |
| `z_offset` | `float` | Vertical position offset. |
| `roll` | `float` | Roll rotation in degrees. |
| `pitch` | `float` | Pitch rotation in degrees. |
| `yaw` | `float` | Yaw rotation in degrees. |

---

## MotionHandle

Returned by `PolicyHandle.add_motion()` and `add_motion_wandb()`.

### MotionHandle.set_metadata

```python
def set_metadata(key: str, value: Any) -> MotionHandle
```

Set a metadata entry for the motion. Returns `self` for chaining.

### MotionHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the motion. |

---

## ViewerConfig

```python
@dataclass
class mjswan.ViewerConfig(
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance: float = 4.0,
    fovy: float | None = None,
    elevation: float = -30.0,
    azimuth: float = 45.0,
    origin_type: OriginType = OriginType.AUTO,
    entity_name: str | None = None,
    body_name: str | None = None,
    env_idx: int = 0,
    max_extra_envs: int = 2,
    enable_reflections: bool = True,
    enable_shadows: bool = True,
    height: int = 240,
    width: int = 320,
)
```

Camera and rendering configuration applied to a scene via `SceneHandle.set_viewer()`. Matches the API of `mjlab.viewer.ViewerConfig`.

**Selected fields**

| Field | Description |
|---|---|
| `lookat` | Look-at point in MuJoCo coordinates (x forward, y left, z up). |
| `distance` | Distance from the look-at point to the viewer. |
| `elevation` | Elevation in degrees (negative = viewer above the look-at point). |
| `azimuth` | Azimuth in degrees from the x-axis (forward), CCW. |
| `fovy` | Vertical field of view in degrees (default 45). |
| `origin_type` | One of `ViewerConfig.OriginType.{AUTO, WORLD, ASSET_ROOT, ASSET_BODY}`. Controls how the camera tracks the scene. |
| `body_name` | Body to track when `origin_type` is `ASSET_BODY`. |
| `enable_reflections` / `enable_shadows` | Toggle three.js reflections and shadows. |

### ViewerConfig.from_position

```python
@staticmethod
def from_position(
    position: tuple[float, float, float],
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    fovy: float | None = None,
    origin_type: ViewerConfig.OriginType | None = None,
    body_name: str | None = None,
) -> ViewerConfig
```

Build a `ViewerConfig` from explicit Cartesian viewer/target positions — `lookat`, `distance`, `elevation`, and `azimuth` are computed.

---

## Command inputs

Each entry is a leaf in a command-term UI block. The aliases (`Slider`, `Button`, `Checkbox`) are identical to their `*Config` counterparts; pick whichever reads better.

### Slider

```python
mjswan.Slider(
    name: str,
    label: str,
    range: tuple[float, float] = (-1.0, 1.0),
    default: float = 0.0,
    step: float = 0.01,
    enabled_when: str | None = None,
)
```

Continuous range slider.

| Field | Description |
|---|---|
| `name` | Internal key used by the policy observation. |
| `label` | Human-readable label shown in the UI. |
| `range` | `(min, max)` bounds. |
| `default` | Initial value. |
| `step` | Slider increment. |
| `enabled_when` | Optional sibling input name that enables this slider (greys it out when the named input is off). |

### Button

```python
mjswan.Button(name: str, label: str)
```

Momentary push button. Fields: `name`, `label`.

### Checkbox

```python
mjswan.Checkbox(name: str, label: str, default: bool = False)
```

Boolean toggle. Fields: `name`, `label`, `default`.

---

## Command helpers

### ui_command

```python
mjswan.ui_command(inputs: list[CommandInput]) -> CommandTermConfig
```

Build a `CommandTermConfig` whose value is driven by manual UI inputs (sliders, buttons, checkboxes). Pass the result to `add_policy(commands={...})`.

```python
target_cmd = mjswan.ui_command(
    [
        mjswan.Slider(
            "target_height", "Target Height (m)", range=(0.3, 1.8), default=1.0
        ),
    ]
)
scene.add_policy(name="PD", policy=model, commands={"target": target_cmd})
```

### velocity_command

```python
mjswan.velocity_command(
    lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z: tuple[float, float] = (-1.0, 1.0),
    default_lin_vel_x: float = 0.5,
    default_lin_vel_y: float = 0.0,
    default_ang_vel_z: float = 0.0,
) -> CommandTermConfig
```

Build a standard `"velocity"` command group (three sliders: `lin_vel_x`, `lin_vel_y`, `ang_vel_z`). Pass it via `commands={"velocity": mjswan.velocity_command(...)}` to `add_policy()`.

### register_command

```python
mjswan.register_command(mjlab_name: str, spec: CommandBinding) -> None
```

Register an adapter from a custom mjlab `*CommandCfg` class to a browser-side command term. `mjlab_name` should typically be the mjlab config class name (e.g. `"LiftingCommandCfg"`).

---

## MDP extension registries

Every observation / termination / event term is normally traced to ONNX from its own Python function — mjswan ships no built-in TypeScript term classes, so nothing needs registering to work. These three override what one mjlab name resolves to when tracing it as-authored will not do:

```python
mjswan.register_observation(name: str, func: ObservationBinding | Callable) -> None
mjswan.register_termination(name: str, func: TerminationBinding) -> None
mjswan.register_event(name: str, func: EventBinding) -> None
```

Two things to register:

- **A trace-friendly replacement callable** (observations only) — same signature as the original, written so `torch.onnx.export` can follow it. What to reach for when the task's own function is correct but not exportable as written (tensor-method RNG, data-dependent control flow).
- **A `*Binding`** — the escape hatch for a term ONNX tracing cannot express at all. `ts_src` is the absolute path of a `.ts` file exporting the class named by `ts_name`; the builder injects it into the browser bundle. A binding without `ts_src` fails the build: mjswan has no built-in class to fall back on.

A term that fails to trace and has neither of these fails the build, with a message naming both options. It is never silently dropped — a missing observation shortens the vector the policy was trained on, and a missing termination or event leaves the browser without a reset condition the task is configured to have.

---

## MjswanApp

Returned by `Builder.build()`.

### MjswanApp.launch

```python
def launch(
    *,
    host: str = "localhost",
    port: int = 8080,
    open_browser: bool = True,
    height: int = 600,
) -> None
```

Start a local HTTP server and (optionally) open the application in a browser. When running inside Google Colab, an inline iframe is rendered instead of starting a blocking server.

The server automatically sets `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers — required for `SharedArrayBuffer`, which MuJoCo WASM uses for threading.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"localhost"` | Bind address (ignored in Colab). |
| `port` | `int` | `8080` | Port. If already in use, the next available port is chosen automatically. |
| `open_browser` | `bool` | `True` | Open the default browser on start (ignored in Colab). |
| `height` | `int` | `600` | Colab iframe height in pixels (ignored outside Colab). |

Blocks until interrupted with `Ctrl-C`.

---

## Output structure

`builder.build()` writes a fully static site:

```
dist/
├── index.html
├── logo.svg
├── manifest.json
├── robots.txt
├── assets/
│   ├── config.json          ← project / scene / policy / splat manifest
│   └── …                    ← compiled JS / CSS
├── _headers                 ← only when Builder(mt=True)
├── coi-serviceworker.js     ← only when Builder(mt=True)
└── <project-id>/            ← "main" for the first project
    ├── index.html
    ├── logo.svg
    ├── manifest.json
    └── assets/
        └── <scene-id>/
            ├── scene.mjz    ← or scene.mjb (depending on add_scene argument)
            ├── <policy-id>.onnx
            ├── <policy-id>.json   ← present when config_path / commands / observations / actions / terminations are set
            ├── <motion-id>.npz       ← one per distinct clip in the scene, shared by its policies
            └── <splat-id>.spz     ← only when source= is used
```

Copy `dist/` to any static host (GitHub Pages, Netlify, S3, …) and it works without a server.
