---
icon: octicons/file-code-16
---

# Building a Policy Config

A policy attached to a scene needs more than just an ONNX file — the browser runtime has to know which observations to feed in, how to interpret the action, what commands to expose in the UI, and when to reset the episode. mjswan exposes all of that as Python kwargs on `add_policy()`, modelled after [mjlab](https://github.com/mujocolab/mjlab){:target="_blank"}'s config classes.

This page is the practical reference for those kwargs. For a runnable end-to-end policy built entirely in Python — including a hand-crafted ONNX graph — see [examples/tutorial/minimum_policy.py](https://github.com/ttktjmt/mjswan/blob/main/examples/tutorial/minimum_policy.py){:target="_blank"}.

## Top level: `add_policy(...)`

```python
scene.add_policy(
    name="Locomotion",
    policy=onnx.load("locomotion.onnx"),
    policy_joint_names=["FL_hip", "FL_thigh", "FL_calf", ...],
    default_joint_pos=[0.1, 0.8, -1.5, ...],
    observations=ObservationGroupCfg(terms={...}),
    actions={"joint_pos": JointPositionActionCfg(...)},
    commands={"velocity": mjswan.velocity_command()},
    terminations={"time_out": TerminationTermCfg(func=term_fns.time_out)},
)
```

The relevant kwargs (see the [API reference](../api/core.md#scenehandleadd_policy) for the full list):

| Kwarg | Purpose |
|---|---|
| `policy_joint_names` | Ordered list of joint names the policy controls. Required for browser-side actuator mapping. |
| `default_joint_pos` | Default pose, one entry per `policy_joint_names`. Used when `use_default_offset=True` on the action term and when an observation subtracts the default pose. |
| `observations` | A single `ObservationGroupCfg`, or a `dict` of them keyed by ONNX input tensor name. Prefer the single group — see below. |
| `actions` | `dict[str, ActionTermCfg]` keyed by term name (e.g. `"joint_pos"`). |
| `commands` | `dict[str, CommandTermConfig]` keyed by policy-visible command name. |
| `terminations` | `dict[str, TerminationTermCfg]` keyed by termination name. |
| `encoder_bias` | Optional per-joint bias; the browser writes `processed_action - encoder_bias` to the actuators (mirrors mjlab). |
| `clip_actions` | Symmetric bound on the raw policy output, applied before any action term, mirroring rsl-rl's `RslRlVecEnvWrapper`. `add_policy_wandb` fills it in from the task's runner config. Not `ActionTermCfg.clip` — see [Actions](#actions). |
| `extras` | Arbitrary JSON payload merged verbatim into the generated policy config. |

## Defaults from an mjlab env config

On a scene from [`add_scene_mjlab`](../getting-started/mjlab.md), `observations` / `commands` / `actions` / `terminations` each default to the matching field of the task's `env_cfg`, so a policy from an mjlab task needs none of them spelled out. Pass one to override that field only; pass `{}` to say the policy genuinely has none. `clip_actions` likewise defaults to the task's runner config.

Per-policy `env_cfg=` takes a different config for one policy — in mjlab terms, several env configs sharing one `scene`, since an env has exactly one observation design. Its control rate must match the scene's `control_dt`: the runtime derives its physics substep count and every timer from one value per scene, so a mismatch raises rather than being silently reinterpreted.

A scene from plain `add_scene` has no config to fall back on, so there every field means exactly what it says.

## Observations

A group is an ordered dict of `ObservationTermCfg` — the runtime concatenates term outputs in declaration order. Pass the group directly:

```python
from mjswan.envs.mdp import observations as obs_fns
from mjswan.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)

obs = ObservationGroupCfg(
    terms={
        "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=obs_fns.projected_gravity_isaac),
        "joint_pos": ObservationTermCfg(
            func=obs_fns.joint_positions_isaac,
            scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=obs_fns.joint_vel_rel,
            scale=0.05,
        ),
        "last_action": ObservationTermCfg(func=obs_fns.previous_actions),
        "velocity_cmd": ObservationTermCfg(
            func=obs_fns.generated_commands,
            params={"command_name": "velocity"},
        ),
    },
)
```

### Why not a dict of groups?

`observations` also accepts `dict[str, ObservationGroupCfg]`, but the keys are **ONNX input tensor names**, not labels — the runtime feeds each group's vector as the input of that name, and `in_keys` defaults to `["policy"]`. An ONNX policy exported by mjlab has exactly one input, so the dict form buys nothing and costs a silent failure mode: a group under the wrong key produces a console warning and a policy that never acts. Passing the group itself lets mjswan pick the key.

The dict form is for the rare multi-input policy, where the config's `in_keys` names each input. Groups named for a training-only mjlab network (`"critic"`) are dropped with a warning: only the actor is exported to ONNX, so nothing would consume them, and leaving them in would trace, bundle, and evaluate them every control step for a value nothing reads.

### Coming from mjlab

An mjlab `env_cfg.observations` is keyed by *network* name, not by ONNX input name — two namespaces that happen to look alike. Hand the whole thing over and mjswan picks the policy's group:

```python
observations = env_cfg.observations  # {"actor": ..., "critic": ...}
observations = env_cfg.observations["actor"]  # or just the group; same result
```

Which group that is comes from the task's **runner** config (`rl_cfg.obs_groups["actor"]`), so a task free to name its groups something else still resolves correctly; the `"actor"` name is only the fallback when there is no registered task to ask. If a task's actor reads *several* groups concatenated, mjswan raises — it feeds one vector per ONNX input and cannot join them, and quietly taking the first would feed the policy a short observation.

A dict whose keys are not mjlab network names is left exactly as written, which is what keeps a policy whose input really is called `"observation"` or `"obs_history"` working.

`ObservationTermCfg` fields used at runtime: `func` (a built-in sentinel below or a custom one registered via `register_observation`), `params` (forwarded to the browser-side class), `scale`, `clip`, `history_length`. Other mjlab fields (`noise`, `delay_*`) are accepted for config compatibility but ignored — there's no training in the browser.

### Built-in observation sentinels

Defined in `mjswan.envs.mdp.observations`:

| Sentinel | Runtime class | Notes |
|---|---|---|
| `base_lin_vel` | `BaseLinearVelocity` | Linear velocity of the base, base frame. |
| `base_ang_vel` | `BaseAngularVelocity` | Angular velocity of the base, base frame. |
| `projected_gravity` | `ProjectedGravityB` | Gravity in the base frame (legacy implementation). |
| `projected_gravity_isaac` | `ProjectedGravity` | Isaac-compatible; defaults to `joint_name="floating_base_joint"`. |
| `joint_pos_rel` | `JointPos` | Joint positions − default pose. |
| `joint_vel_rel` | `JointVelocities` | Joint velocities − default velocities. |
| `joint_positions_isaac` | `JointPositions` | Isaac joint ordering, default-subtracted. |
| `last_action` | `PrevActions` | Most recent action tensor. |
| `previous_actions` | `PreviousActions` | Isaac-compatible most-recent action tensor. |
| `generated_commands` | `GeneratedCommands` | Requires `params={"command_name": "<name>"}`. |
| `velocity_command_with_oscillators` | `VelocityCommandWithOscillators` | 16-dim velocity command + oscillator signals. |
| `impedance_command` | `ImpedanceCommand` | Impedance command tensor. |
| `joint_pos_cos_sin` | `JointPosCosSin` | `[cos(q), sin(q)]` for one joint. |
| `motion_anchor_pos_b` | `MotionAnchorPosB` | Tracking: anchor position in the robot anchor frame. |
| `motion_anchor_ori_b` | `MotionAnchorOriB` | Tracking: anchor orientation in the robot anchor frame. |
| `robot_body_pos_b` | `RobotBodyPosB` | Tracking: robot body positions in the robot anchor frame. |
| `robot_body_ori_b` | `RobotBodyOriB` | Tracking: robot body orientations in the robot anchor frame. |
| `builtin_sensor` | `BuiltinSensor` | Raw data from a named MuJoCo sensor. |

For a custom observation backed by your own TypeScript class, see `register_observation` in the API reference.

## Actions

Each entry in `actions` is a subclass of `ActionTermCfg`. Two are supported in the browser:

```python
from mjswan.envs.mdp.actions import (
    JointPositionActionCfg,
    JointEffortActionCfg,
)

# Joint-position control with external PD
actions = {
    "joint_pos": JointPositionActionCfg(
        actuator_names=(".*",),
        scale=0.25,
        offset=0.0,
        use_default_offset=True,  # action=0 commands the default pose
        stiffness=40.0,  # kp (scalar, per-joint list, or dict by joint name)
        damping=1.0,  # kd
    ),
}

# Direct torque output
actions = {
    "thrust": JointEffortActionCfg(
        actuator_names=("lift",),
    ),
}
```

`stiffness` and `damping` are mjswan-specific — in mjlab they live on the actuator, but the browser runtime computes PD externally for motor actuators with `biastype=none`, so we need them in the policy config. Both accept a scalar, a per-joint list (aligned with `policy_joint_names`), or a dict keyed by joint name.

`JointVelocityActionCfg`, `TendonLengthActionCfg`, `TendonVelocityActionCfg`, `TendonEffortActionCfg`, and `SiteEffortActionCfg` are exported so mjlab configs import cleanly, but they raise `NotImplementedError` at build time — the browser runtime doesn't support them yet.

### Two clips, and they are not the same bound

| | `ActionTermCfg.clip` | `clip_actions` |
|---|---|---|
| Lives on | the action term | the policy (from mjlab's *runner* config) |
| Shape | `dict` of `{pattern: (min, max)}`, per target | one number, symmetric `[-v, +v]` |
| Applied to | `raw * scale + offset` | the policy's raw output, before any term |
| mjlab source | `BaseAction.process_actions` | rsl-rl's `RslRlVecEnvWrapper.step` |

`clip_actions` lands ahead of everything, so a `last_action` observation reads the clamped vector — matching mjlab, where the wrapper clamps before `env.step` and the action manager records what it was handed. `add_policy_wandb` reads it from the task automatically; pass it explicitly only for a hand-built policy or with `only_latest=True`, which skips mjlab entirely.

## Commands

`commands` is a dict of `CommandTermConfig` values keyed by the policy-visible name your observations reference (e.g. `params={"command_name": "velocity"}` looks up `commands["velocity"]`).

```python
commands = {
    "velocity": mjswan.velocity_command(),  # standard 3-DoF locomotion
    "target": mjswan.ui_command(
        [  # arbitrary UI inputs
            mjswan.Slider(
                "target_height", "Target Height (m)", range=(0.3, 1.8), default=1.0
            ),
        ]
    ),
}
```

To adapt a custom mjlab command class to a browser-side TS class, use `mjswan.register_command(mjlab_name, spec)` — see the [API reference](../api/core.md#register_command).

## Terminations

Same shape as observations:

```python
from mjswan.envs.mdp import terminations as term_fns
from mjswan.managers.termination_manager import TerminationTermCfg

terminations = {
    "time_out": TerminationTermCfg(func=term_fns.time_out, time_out=True),
    "fallen": TerminationTermCfg(
        func=term_fns.bad_orientation,
        params={"limit_angle": 1.0},
    ),
}
```

### Supported termination sentinels

| Sentinel | Runtime class | Required params |
|---|---|---|
| `time_out` | `TimeOut` | — |
| `bad_orientation` | `BadOrientation` | `limit_angle` (radians) |
| `root_height_below_minimum` | `RootHeightBelowMinimum` | `minimum_height` (metres) |

Other mjlab termination sentinels are exported for config compatibility but raise `NotImplementedError` at build time. Register a custom one with `register_termination`.

## End-to-end examples

| Example | What it shows |
|---|---|
| [examples/tutorial/minimum_policy.py](https://github.com/ttktjmt/mjswan/blob/main/examples/tutorial/minimum_policy.py){:target="_blank"} | Smallest possible policy — a hand-built ONNX PD controller plus observations, actions, and a `ui_command`, all in one file. |
| [examples/demo/gentle_humanoid/main.py](https://github.com/ttktjmt/mjswan/blob/main/examples/demo/gentle_humanoid/main.py){:target="_blank"} | Realistic tracking policy with `JointPositionActionCfg`, motions attached via `add_motion(...)`, and per-motion metadata. |

## Legacy: passing a JSON file via `config_path=`

`add_policy(config_path="policy.json")` still accepts a hand-written JSON file. mjswan reads it, merges the Python-side `commands` / `observations` / `actions` / `terminations` into it, and writes the result alongside the ONNX. Prefer the Python kwargs above for new policies — the JSON form is mostly useful when importing a config that was already produced by another tool.

For the on-disk JSON schema, run `git log -- docs/docs/notes/policy-config.md` and read the version prior to this rewrite.
