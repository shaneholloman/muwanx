---
icon: octicons/code-16
---

# Examples

Copy-paste patterns for common use cases. For a step-by-step first run, see [Quickstart](quickstart.md).

## Scene from an XML string

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Demo")

spec = mujoco.MjSpec.from_string("""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
""")
project.add_scene(spec=spec, name="Sphere")

builder.build().launch()
```

## Scene from a file

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Robot")
project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="My Robot",
)
builder.build().launch()
```

## Policy with velocity command sliders

```python
import mujoco
import onnx
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Robot")

scene = project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="G1",
)
scene.add_policy(
    name="Locomotion",
    policy=onnx.load("robot/locomotion.onnx"),
    config_path="robot/locomotion.json",
    commands={
        "velocity": mjswan.velocity_command(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            default_lin_vel_x=0.5,
        )
    },
)

builder.build().launch()
```

`config_path` points to a JSON file describing the observation and action convention. See [Policy Config Format](../notes/policy-config.md) for the schema.

## Multiple policies on one scene

```python
scene = project.add_scene(spec=spec, name="Go2")

scene.add_policy(
    name="Policy A",
    policy=onnx.load("policy_a.onnx"),
    config_path="policy_a.json",
    commands={"velocity": mjswan.velocity_command()},
)

scene.add_policy(
    name="Policy B",
    policy=onnx.load("policy_b.onnx"),
    config_path="policy_b.json",
    commands={"velocity": mjswan.velocity_command()},
)
```

The browser UI shows a selector for choosing between policies at runtime.

## Custom command inputs

Pass a `commands={...}` dict to `add_policy()`. Each value is a `CommandTermConfig` built with `mjswan.ui_command([...])` (manual UI) or `mjswan.velocity_command(...)` (locomotion shortcut).

```python
scene.add_policy(
    name="Locomotion",
    policy=onnx.load("robot/locomotion.onnx"),
    config_path="robot/locomotion.json",
    commands={
        "velocity": mjswan.ui_command(
            [
                mjswan.Slider(
                    "lin_vel_x", "Forward", range=(-2.0, 2.0), default=0.5, step=0.05
                ),
                mjswan.Slider(
                    "lin_vel_y", "Lateral", range=(-0.5, 0.5), default=0.0, step=0.05
                ),
                mjswan.Slider(
                    "ang_vel_z", "Yaw Rate", range=(-1.0, 1.0), default=0.0, step=0.05
                ),
            ]
        ),
    },
)
```

See [examples/tutorial/minimum_policy.py](https://github.com/ttktjmt/mjswan/blob/main/examples/tutorial/minimum_policy.py){:target="_blank"} for a full example that builds a policy plus its observation / action / command config entirely from Python — no `config_path` required.

## Multiple projects

Each `add_project()` call maps to its own URL. The first project is always the root (`/`); the rest live at `/<id>/`.

```python
builder = mjswan.Builder(base_path="/demo/")

quadrupeds = builder.add_project(name="Quadrupeds")
quadrupeds.add_scene(spec=mujoco.MjSpec.from_file("go2/scene.xml"), name="Go2")
quadrupeds.add_scene(spec=mujoco.MjSpec.from_file("go1/scene.xml"), name="Go1")

humanoids = builder.add_project(name="Humanoids", id="humanoid")
humanoids.add_scene(spec=mujoco.MjSpec.from_file("g1/scene.xml"), name="G1")

builder.build().launch()
```

## Gaussian Splat background (bundled)

Use `source=` to bundle a `.spz` file into the built application. This is the recommended approach: the file is copied into `dist/` at build time, so the app works offline.

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Robot")

scene = project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="G1",
)
scene.add_splat(
    "Lab",
    source="lab.spz",  # bundled into dist/
    scale=1.35,
    z_offset=0.71,
)

builder.build().launch()
```

## Gaussian Splat background (external URL)

Use `url=` to reference a `.spz` file hosted externally. The build stays small, but the browser must fetch the file at runtime.

```python
scene.add_splat(
    "Outdoor",
    url="https://example.com/outdoor.spz",
    scale=3.0,
    z_offset=0.5,
)
```

## Multiple splats on one scene

Add several splats to the same scene — the viewer shows a selector to switch between them at runtime.

```python
scene.add_splat("Lab A", source="lab_a.spz", scale=1.35, z_offset=0.71)
scene.add_splat("Lab B", source="lab_b.spz", scale=1.20, z_offset=0.65)
```

## Splat URL input without pre-configured splats

Call `enable_splat_section()` to show the Splat selector in the control panel even when no splats are pre-configured. Users can then paste an arbitrary `.spz` URL at runtime.

```python
scene = project.add_scene(name="Demo", spec=spec)
scene.enable_splat_section()
```

## Calibrating a new splat capture

Set `control=True` to expose live sliders for scale, offset, and rotation while you dial in the alignment. Remove `control=True` once the values are finalised.

```python
scene.add_splat(
    "Lab",
    source="lab.spz",
    scale=1.35,
    z_offset=0.71,
    control=True,  # shows calibration controls in the viewer
)
```

## Headless build (no browser)

```python
import os

app = builder.build()
# dist/ is ready; app.launch() is not called
```

Or without modifying the script:

```bash
python build.py
```

See [Deployment](../guides/deployment.md) for GitHub Pages and CI/CD setup.
