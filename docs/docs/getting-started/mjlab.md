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

The W&B form requires both `mjlab` and `torch` (the `model_*.pt` → ONNX conversion runs locally).

Each attached policy configures itself from the task: its observations, commands, actions and terminations all come from the task's `env_cfg`, and its raw-action bound from the task's runner config. For finer control, drop down to the next two patterns.

## 2. Scene helper: `ProjectHandle.add_scene_mjlab`

When you need multiple scenes (one per task) or want to mix mjlab tasks with hand-written scenes, use `add_scene_mjlab(task_id)` on a `ProjectHandle`. It loads the task's MuJoCo spec, applies the task's `viewer` / `events` / terrain data, and returns a normal `SceneHandle`.

It loads mjlab's **play** config by default — the opposite of mjlab's own `load_env_cfg`, because that default serves training scripts and this is a playback tool. The training config sets `episode_length_s` to 10–20 s, and mjswan turns that into the browser's `time_out` termination, so a viewer built from it resets the robot every few seconds. Pass `play=False` if you want training-time conditions.

```python
import mjswan
from mjlab.tasks.registry import list_tasks

builder = mjswan.Builder()
project = builder.add_project(name="mjlab Tasks")

for task_id in list_tasks():
    project.add_scene_mjlab(task_id)

builder.build().launch()
```

### Attaching trained policies from W&B

Use `scene.add_policy_wandb(run_path)` to fetch checkpoints from one or more W&B runs and attach them all to the scene:

```python
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="ANYmal C")

scene = project.add_scene_mjlab("Mjlab-Velocity-Flat-Anymal-C")
scene.add_policy_wandb("<entity>/<project>/<run_id>")

builder.build().launch()
```

The scene keeps the `env_cfg` it was built from, and every policy added to it defaults its observations, commands, actions and terminations to that config — so the run path is all you need. `task_id` also defaults to the scene's task.

### Editing the config first

Some tasks are incomplete as registered: mjlab's tracking tasks ship `commands["motion"].motion_file = ""` for the caller to fill in. Load the config, edit it, and pass it to `add_scene_mjlab` — the policies then inherit your edited version, because the scene holds the same object:

```python
from mjlab.tasks.registry import load_env_cfg

task_id = "Mjlab-Tracking-Flat-Unitree-G1"
# `play=True` here, not on `add_scene_mjlab`: passing `env_cfg` means the scene uses it
# as given, so the choice of config is already made by the time it gets there.
env_cfg = load_env_cfg(task_id, play=True)
env_cfg.commands["motion"].motion_file = "artifacts/spinkick.npz"

scene = project.add_scene_mjlab(task_id, env_cfg=env_cfg)
scene.add_policy_wandb("<entity>/<project>/<run_id>")
```

Load it **once**. `load_env_cfg` returns a deepcopy, so a second call gives you an equal but separate config, and edits to one are invisible to the other.

### Overriding one field

Pass any of the four to replace just that one; the rest still come from the config. Pass `{}` to say the policy genuinely has none:

```python
scene.add_policy_wandb(
    "<entity>/<project>/<run_id>",
    terminations={"time_out": TerminationTermCfg(func=term_fns.time_out)},
)
```

mjswan adapts mjlab config classes automatically (mjlab is a soft dependency). `observations` also accepts the task's whole `env_cfg.observations` dict or a single group — see [Policy Config Format](../notes/policy-config.md), which explains why the key matters.

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

# `play=True` for the same reason `add_scene_mjlab` defaults to it: this is a viewer.
env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Anymal-C", play=True)
env_cfg.scene.num_envs = 1  # single environment for the viewer
scene_obj = Scene(env_cfg.scene, device="cpu")

project.add_scene(
    spec=scene_obj.spec,  # MjSpec from mjlab
    name="ANYmal C",
)

builder.build().launch()
```
