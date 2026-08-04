---
icon: octicons/arrow-switch-16
---

# Using mjlab

[mjlab](https://github.com/mujocolab/mjlab){:target="_blank"} is a reinforcement learning framework built on top of MuJoCo. mjswan can visualize mjlab environments directly — there is no need to export or convert anything.

This page walks through three integration levels, from the one-line shortcut to the full manual form.

## 1. One-liner: `Builder.from_mjlab`

The fastest path. `Builder.from_mjlab(task_id, run_path=...)` creates a project, adds the mjlab scene, and (optionally) attaches every `model_*.pt` checkpoint from one or more W&B runs as ONNX policies.

```python
import mjswan

# Just visualize the scene
app = mjswan.Builder.from_mjlab("go2_flat").build()
app.launch()

# Visualize the scene + every checkpoint from a W&B run
app = mjswan.Builder.from_mjlab(
    "Mjlab-Velocity-Flat-Anymal-C",
    run_path="<entity>/<project>/<run_id>",
).build()
app.launch()
```

The W&B form requires both `mjlab` and `torch` (the `model_*.pt` → ONNX conversion runs locally). For finer control, drop down to the next two patterns.

## 2. Scene helper: `ProjectHandle.add_scene_mjlab`

When you need multiple scenes (one per task) or want to mix mjlab tasks with hand-written scenes, use `add_scene_mjlab(task_id, play=...)` on a `ProjectHandle`. It loads the task's MuJoCo spec, applies the task's `viewer` / `events` / terrain data, and returns a normal `SceneHandle`.

```python
import mjswan
from mjlab.tasks.registry import list_tasks

builder = mjswan.Builder()
project = builder.add_project(name="mjlab Tasks")

for task_id in list_tasks():
    project.add_scene_mjlab(task_id, play=True)

builder.build().launch()
```

### Attaching trained policies from W&B

Use `scene.add_policy_wandb(run_path, task_id=..., ...)` to fetch checkpoints from one or more W&B runs and attach them all to the scene. Pass `observations` / `commands` / `actions` / `terminations` from the mjlab `env_cfg` — mjswan adapts mjlab config classes automatically.

```python
import mjswan
from mjlab.tasks.registry import load_env_cfg

builder = mjswan.Builder()
project = builder.add_project(name="ANYmal C")

task_id = "Mjlab-Velocity-Flat-Anymal-C"
env_cfg = load_env_cfg(task_id, play=True)

scene = project.add_scene_mjlab(task_id, play=True)
scene.add_policy_wandb(
    "<entity>/<project>/<run_id>",
    task_id=task_id,
    commands=env_cfg.commands,
    actions=env_cfg.actions,
    terminations=env_cfg.terminations,
)

builder.build().launch()
```

`add_policy_wandb` accepts a `list[str]` for the run path if you want to bundle checkpoints from multiple runs together. The latest checkpoint (highest training step) is marked as the default.

If you only want the exported `.onnx` artifact (skipping the `.pt → .onnx` conversion), pass `only_latest=True` — `task_id` is then optional.

## 3. Full manual form

For maximum control — e.g. customising `env_cfg` before building the scene — fall back to mjlab's own `Scene` class and pass `spec` to `add_scene` directly:

```python
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="mjlab Examples")

env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Anymal-C")
env_cfg.scene.num_envs = 1  # single environment for the viewer
scene_obj = Scene(env_cfg.scene, device="cpu")

project.add_scene(
    spec=scene_obj.spec,  # MjSpec from mjlab
    name="ANYmal C",
)

builder.build().launch()
```
