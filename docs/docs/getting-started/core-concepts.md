---
icon: octicons/light-bulb-16
---

# Core Concepts

mjswan uses a four-level hierarchy to describe a browser application: **Builder → Project → Scene → Policy/Splat**. Understanding this structure is the fastest way to get oriented before looking at the API.

```
Builder
  └── Project
        └── Scene
              ├── Splat
              └── Policy
                    └── Motion
```

## Builder

`Builder` is the entry point. It collects everything you define and, when you call `build()`, compiles it into a self-contained web application written to `dist/` (or a directory you choose).

```python
import mjswan

builder = mjswan.Builder()
# … add projects …
app = builder.build()
app.launch()
```

Two optional constructor arguments matter for deployment:

| Argument | Default | Purpose |
|---|---|---|
| `base_path` | `"/"` | URL prefix when hosting at a subdirectory (e.g. `"/mjswan/"` for a GitHub Pages project page) |
| `gtm_id` | `None` | Google Tag Manager container ID; injects the GTM snippet when set |

## Project

A project groups related scenes under a single URL. The first project added becomes the root (`/`); additional projects are reachable at `/<project-id>/`.

```python
project = builder.add_project(name="My Robots")

# Explicit URL slug — accessible at /demo/
demo = builder.add_project(name="Demo", id="demo")
```

If you omit `id`, mjswan derives it automatically from the project name (spaces and hyphens become underscores, lowercased). The first project is always the root regardless of `id`.

## Scene

A scene contains exactly one MuJoCo model. You supply either an `MjSpec` or an `MjModel`:

```python
import mujoco

# Compressed .mjz — smaller output, recommended for large deployments
scene = project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="My Robot",
)

# Binary .mjb — loads slightly faster in the browser, produces larger files
scene = project.add_scene(
    model=mujoco.MjModel.from_xml_path("robot/scene.xml"),
    name="My Robot",
)
```

!!! tip "Which format should I use?"
    Use `spec=` unless you have a specific reason to prefer `model=`. The `.mjz` format uses DEFLATE compression and is significantly smaller — important when approaching GitHub Pages' 1 GB deployment limit.

## Splat

A splat is a [Gaussian Splat](https://en.wikipedia.org/wiki/Gaussian_splatting) background rendered behind the MuJoCo simulation. Splats are stored as `.spz` files and give scenes a photorealistic real-world environment without affecting physics.

Add one or more splats to a scene using `add_splat()`. You must supply exactly one of `source` or `url`:

```python
# Recommended: bundle the .spz file into the app
scene.add_splat(
    "Lab Environment",
    source="lab.spz",  # copied into dist/ at build time
    scale=1.35,  # converts splat units → meters
    z_offset=0.71,  # vertical shift to align ground planes
)

# Alternative: reference an external URL (not bundled)
scene.add_splat(
    "Outdoor",
    url="https://example.com/outdoor.spz",
    scale=3.0,
    z_offset=0.5,
)
```

When multiple splats are attached to the same scene, the viewer shows a selector so users can switch between them at runtime.

### Source vs URL

| Option | Effect |
|---|---|
| `source` | Copies the `.spz` into `dist/` at build time — fully self-contained, works offline |
| `url` | Browser fetches the file at runtime — smaller build, requires network access |

### Alignment controls

| Parameter | Description |
|---|---|
| `scale` | Metric scale factor (splat units → metres). Use `metric_scale_factor` from capture metadata if available |
| `x_offset`, `y_offset`, `z_offset` | Position offsets in scaled splat units. `z_offset` aligns ground planes; use `ground_plane_offset` from capture metadata if available |
| `roll`, `pitch`, `yaw` | Rotation in degrees applied on top of the COLMAP → Three.js base rotation |

Set `control=True` to expose these alignment controls as live sliders in the viewer — useful while calibrating a new capture:

```python
scene.add_splat("Lab", source="lab.spz", scale=1.35, control=True)
```

### Splat selector without pre-configured splats

By default, the Splat selector only appears when at least one splat is attached to the scene. Call `enable_splat_section()` to show the selector unconditionally — this lets viewers paste an arbitrary `.spz` URL directly in the control panel at runtime:

```python
scene.enable_splat_section()
```

## Policy

A policy is an ONNX model that runs inference inside the browser. Attach one or more policies to a scene:

```python
import onnx

policy = scene.add_policy(
    name="Locomotion",
    policy=onnx.load("locomotion.onnx"),
    config_path="locomotion.json",  # optional: observation/action config
)
```

Policies are purely client-side: inference runs in the browser via onnxruntime-web, so no server is needed at runtime.

You can also build the observation / action / termination config entirely from Python by passing `observations=`, `actions=`, `commands=`, and `terminations=` to `add_policy()` — `config_path` becomes optional in that case. See [examples/tutorial/minimum_policy.py](https://github.com/ttktjmt/mjswan/blob/main/examples/tutorial/minimum_policy.py){:target="_blank"} for a fully-Python example.

### Commands

Commands let users interact with a running policy — for example, steering a walking robot with velocity sliders. Pass a `commands=` dict to `add_policy()`:

```python
scene.add_policy(
    name="Locomotion",
    policy=onnx.load("locomotion.onnx"),
    commands={
        "velocity": mjswan.ui_command(
            [
                mjswan.Slider(
                    "lin_vel_x", "Forward Velocity", range=(-1.0, 1.0), default=0.5
                ),
                mjswan.Slider(
                    "lin_vel_y", "Lateral Velocity", range=(-0.5, 0.5), default=0.0
                ),
                mjswan.Slider("ang_vel_z", "Yaw Rate", range=(-1.0, 1.0), default=0.0),
            ]
        ),
    },
)
```

Available command inputs:

| Class | Description |
|---|---|
| `mjswan.Slider` | Continuous range slider. Fields: `name`, `label`, `range`, `default`, `step` |
| `mjswan.Button` | Momentary push button. Fields: `name`, `label` |
| `mjswan.Checkbox` | Boolean toggle. Fields: `name`, `label`, `default` |

### Motion

Motion-tracking policies need one or more reference motions (`.npz` files) loaded alongside the ONNX model. Attach them to the `PolicyHandle`:

```python
policy = scene.add_policy(name="Tracker", policy=onnx.load("tracker.onnx"))

# Local .npz bundled into dist/ at build time
policy.add_motion(
    name="default",
    source="motions/walk.npz",
    fps=50.0,
    anchor_body_name="pelvis",
    body_names=("pelvis",),
    default=True,
    loop=False,
)

# Or fetch from a W&B run
policy.add_motion_wandb(
    run_path="<entity>/<project>/<run_id>",
    anchor_body_name="pelvis",
    body_names=("pelvis",),
)
```

`anchor_body_name` and `body_names` are required — they tell the browser-side tracker which bodies in the MuJoCo model correspond to the dataset. `default=True` marks the motion as the one selected on load; when multiple motions are attached the viewer shows a selector. See the API reference for the full parameter list.

## Output structure

`builder.build()` writes the following layout to the output directory:

```
dist/
├── index.html
├── logo.svg
├── manifest.json
├── robots.txt
├── assets/
│   ├── config.json          ← project/scene/policy manifest
│   └── …                    ← compiled JS/CSS
├── _headers                 ← only when Builder(mt=True)
├── coi-serviceworker.js     ← only when Builder(mt=True)
└── <project-id>/            ← "main" for the first project
    ├── index.html
    ├── logo.svg
    ├── manifest.json
    └── assets/
        └── <scene-id>/
            ├── scene.mjz    ← or scene.mjb
            ├── <policy>.onnx
            ├── <policy>.json  ← present when config_path / commands / observations / actions / terminations are set
            ├── <motion>.npz    ← one per distinct clip in the scene, shared by its policies
            └── <splat>.spz    ← only when source= is used
```

The result is a fully static site: copy `dist/` to any static host (GitHub Pages, Netlify, S3, …) and it works without a server.

<!-- MEDIA: suggest a screenshot of the browser UI showing scene and policy selector panels -->

## Environment variables

| Variable | Effect |
|---|---|
| `MJSWAN_BASE_PATH` | Read by the Vite build (`vite.config.ts`) and used as the asset base. Useful in CI pipelines. |
| `MJSWAN_NO_LAUNCH` | Convention used by the bundled example scripts (e.g. `examples/demo/main.py`) to skip `app.launch()` after building. Honor it in your own build scripts to make them CI-friendly. |
