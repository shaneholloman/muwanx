# Companion brief: ONNX Command & Event migration

> Companion to [ADR 0005](0005-onnx-traced-terms-superseding-the-declarative-dsl.md).
> This is an **implementation brief**, not a decision record — it turns ADR 0005's
> phased plan into concrete Command/Event migration scope spanning both the Python
> build path and the TypeScript runtime, and it carries a **review checkpoint** of
> findings surfaced by the first real trace runs. Read it alongside the ADR.

## Status

| Area | State |
|---|---|
| §2 tracer on Cartpole (Obs/Term) | **done** — `src/mjswan/compile/tracer.py`, parity clean |
| §2b RNG spy/replay harness | **done** — `src/mjswan/compile/rng.py` |
| Reset-mode Event tracing (`write_joint_state_to_sim`) | **done** — Cartpole `reset_slider`/`reset_hinge` parity clean |
| Dynamic-slot Event path (reads live state) | **done** — Go1 `push_robot` reads live `root_link_vel_w` as a graph input, parity clean |
| `entity_write` — root **velocity** | **done** — Go1 `push_robot` (`write_root_link_velocity_to_sim`), parity clean |
| `entity_write` — root **pose** | **done** — Go1 `reset_base` (`write_root_link_pose_to_sim`), parity clean (max\|Δ\|≈1e-7) |
| Scene-const + control-flow-scalar capture | **done** — `env.scene.env_origins`, `asset.is_fixed_base` baked as constants |
| §3 `OnnxCommand` tracer (Python) | **partial** — stateful `trace_command_term` done; `LiftingCommand` (§3b) parity clean |
| §3b `LiftingCommandCfg` body | **done** — Lift-Cube-Yam, rand_dim=7, mask gate + cube entity_write, parity clean |
| §3a `UniformVelocityCommandCfg` body | **traceable via override** — examples-side trace-friendly override incl. heading tracking traces + parity clean (dynamic-slot Command support, finding 15) |
| Dynamic-slot Command support (runtime reads) | **done** — `term.robot`/`term._env` swapped to the Event tagged-key proxies; `heading_w` threaded as a graph input |
| `OnnxCommand` `policy.json` config output | **done** — `command_config`/`write_command_artifact` + JSON schema; unit-tested + end-to-end from real traces |
| §4 orchestrator-owned seeded PRNG (TS) | **done** — `core/rng.ts` (xoshiro128**, state snapshot for replay) |
| §4 interval/startup/reset Event triggers (TS) | **done** — `core/event/triggers.ts`, scalar per ADR §5 |
| §4 `entity_write` apply primitive (TS) | **done** — `core/event/entityWrite.ts` (joint_state/root_pose/root_velocity) |
| §3 generic `OnnxCommand` handler (TS) | **done** — `core/command/OnnxCommand.ts` (timer, state, rand, UI override, writes) |
| §4 wire triggers/OnnxCommand into EventManager/CommandManager + runtime | not started |
| §3a `SliderCommandConfig` extension (TS) | not started |
| Remove `src/mjswan/dsl/` + `scripts/verify_dsl_migration.py` | **deferred** — see note below |

**Sequencing (dependency order, not a schedule):** RNG harness (§2b) → tracer on
Cartpole → **this review checkpoint** → interval-event dynamic-slot + `entity_write`
generalization (a) → §3 `OnnxCommand` Python side → §4 native timers + entity-write
apply (TS) → §3a `SliderCommandConfig` last. Cartpole was the deliberate first
target: no Command, only `mode="reset"` Events, so it exercises the tracer and RNG
harness in isolation before Command/Event complexity is layered on.

## 1. Reference tasks (`examples/mjlab/`, full set — not a curated pair)

| Project | task_id(s) | Command pattern | Treatment |
|---|---|---|---|
| defaults | Cartpole-Balance / -Swingup | none | full ONNX parity (Obs/Term/Action/reset-Event) |
| defaults | Velocity-Flat/Rough × G1/Go1 | `UniformVelocityCommandCfg` | full parity **incl. Command** (§3a) |
| defaults | Lift-Cube-Yam | `LiftingCommandCfg` → ONNX (§3b) | full parity **incl. Command** |
| g1_spinkick | Tracking-Flat-…-No-State-Estimation | `TrackingCommand` (motion lookup) | **carved out** — open design |
| musclemimic | myoMimicFullbody-v0 | `TrackingCommand`-shaped (Mimic) | **carved out** (private-repo dep) |
| unitree_rl | Unitree-G1-Tracking-… | `TrackingCommand`-shaped | **carved out** |
| myosuite | Myosuite-* | none (scene-only, no policy) | nothing to trace |

**Exit criterion:** for every task, Obs/Term/Action ONNX parity holds using whichever
Event modes that task declares. Cartpole needs only `mode="reset"`. Velocity-Flat/Rough
need `mode="startup"` and `mode="interval"` (`push_robot`); Lift-Cube-Yam needs
`mode="startup"`. Tasks with `UniformVelocityCommandCfg`/`LiftingCommandCfg` must also
match the Command term body (§3a/§3b). `TrackingCommand`-shaped commands keep their
existing native implementation and are **not** required to be ONNX-traced yet. A green
run does **not** validate Command tracing beyond `UniformVelocityCommandCfg` and
`LiftingCommandCfg`.

## 2. Tracer approach

**Discard, do not evolve, `src/mjswan/dsl/`.** The `SymbolicEnv` tracer is a bespoke
operator-overloading recorder over a closed op set; it does not run real tensors and
cannot express anything `torch.onnx.export` can't do better. The term-serialization
step in the Builder/`policy.py` path is rewritten to (1) wrap each term body in a
`torch.nn.Module`, (2) resolve every `SceneEntityCfg` regex to static indices at trace
time (baked as graph constants), (3) trace with `torch.onnx.export` (dynamic batch
axis, static everything else). `scripts/verify_dsl_migration.py` is superseded by the
numeric-parity harness.

> **Deferral note (implementation reality).** The current `config.json`/`policy.json`
> format and the shipping frontend interpreter still consume the DSL graph shape.
> `dsl/` and `verify_dsl_migration.py` are removed **only once the ONNX path actually
> replaces the DSL in the build output** (a later phase), per the ADR's own gate
> ("once the new harness supersedes their coverage"). Removing now would break the
> shipping pipeline.

**On tracing risk:** mjlab must run thousands of parallel envs on GPU, which forces
term bodies to avoid per-env Python `if`/`for` on tensor values in favor of masking
(`torch.where`) — the same shape `torch.onnx.export` needs. So terms are likely close
to traceable by construction. §9's fail-loudly escape hatch (native TS reimplementation,
never force-trace, never ship source) is the safety net.

## 2b. RNG alignment for the parity harness

**Build/test-time only — never ships to the browser.** Unrelated to ADR 0005 §2's
orchestrator-owned seeded PRNG (runtime randomness + bit-for-bit session replay); do
not conflate them. The comparison is `allclose`-within-tolerance, not bit-for-bit
(float32 eager PyTorch vs ONNX Runtime can reorder ops in the last bits). It compares
mjlab's live Python env against the exported graphs run via `onnxruntime` in Python;
the browser runtime (WASM physics, TS orchestrator, ONNX Runtime Web) is exercised only
in later phases.

Because Event/Command bodies take `rand` as an explicit input (ADR §2), the harness
must **record every random draw mjlab makes** during the reference computation and
**replay those exact values** into the graph's `rand` input.

1. **Spy, don't stub.** Patch the term function's own module globals (e.g.
   `reset_joints_by_offset.__globals__["sample_uniform"]`) — not the source module,
   which mjlab has already imported by name — so RNG helpers still return mjlab's real
   draw while recording `(values)` in call order.
2. **Count each term's draws from source, do not guess.** Record the count as
   `rand_dim` in the term's config. (Cartpole reset: 1 pos + 1 vel = 2. `push_robot`
   via `_sample_se3_range`: 6. `LiftingCommand._resample_command`: 3 target pos + 1 yaw
   + 3 cube pos + its orientation draw — enumerate before tracing.)
3. Replay recorded values as `rand`, in consumption order. Non-firing steps
   (`resample_mask=False`) can receive zeros — the graph's `torch.where` discards them.
4. This validates the traced *math* reproduces mjlab's transform of a given draw into a
   final value. It does **not** validate that mjlab and mjswan draw the same numbers at
   runtime (separate, already-solved concern — ADR §2).
5. Applies equally to Event terms with randomness — `push_robot`, and the startup DR
   terms (`foot_friction`, `encoder_bias`, `base_com`) — not just Command.

Resample *timing* (native trigger deciding when `resample_mask` flips) is a TS-side
concern, out of scope for this Python harness — it only needs to exercise both
`resample_mask` states directly.

## 3. Command designs — one shared mechanism

The engine ships **one** generic built-in, `OnnxCommand`:

```
forward(prev_state, resample_mask, rand, robot_state) -> (next_state, command, entity_write?)
```

It owns the resample timer (scalar, ADR §5), persists `state` across frames (reusing
`OnnxModule`'s `is_init`/`carry` convention, §4), calls the term's `.onnx` each frame,
optionally overwrites `command` from a declarative `ui` config (§3a), and — if the graph
emits `entity_write` — applies it via a generic "write a named entity's pose/velocity"
primitive.

> **`entity_write` is new work, not reuse.** `DslEvent.ts`'s `Mutation`/`Sample`
> machinery *samples and writes in one step* (`Math.random()` then apply); it has no
> "apply an already-computed value" entry point. Its low-level mjData writes (joint
> qpos/qvel, freejoint pose) may be reusable building blocks, but the "take an
> ONNX-produced value and write it to a named entity's field" primitive must be built.

`UniformVelocityCommandCfg` and `LiftingCommandCfg` are **data instantiations** of
`OnnxCommand` — different `.onnx`, `state_fields`, `ui`/`entity_write` — no engine-side
class each. `register_command`/`CommandBinding` keeps its shape but `ts_name` points at
`OnnxCommand` and `serializer` traces to ONNX. `LiftingCommand.ts` and
`_serialize_uniform_velocity_command`'s UI-only mapping are superseded. `UiCommand`
stays for genuinely never-computed commands.

### 3a. `UniformVelocityCommandCfg` → `OnnxCommand` config

**Match mjlab play-time exactly** (`velocity_command.py`, `scripts/play.py`,
`viewer/viser/viewer.py`). Key property: **the autonomous computation is never
skipped** — `compute()` runs `super().compute()` (resample + heading tracking) every
frame regardless of GUI, then the GUI *conditionally overwrites* per-axis for the viewed
env only. `apply_gui_reset`/`on_viewer_pause` are base no-ops for this type (keep them
as generic no-op interface points).

State fields (each needs a declared **shape + dtype** in `policy.json`, not just a name):
`vel_command_b` `(3,) f32`, `vel_command_w` `(3,) f32`, `heading_target` `() f32`,
`is_heading_env` `() bool`, `is_standing_env` `() bool`, `is_world_env` `() bool`,
`is_forward_env` `() bool`. No `entity_write` for the common case. `ui`: one checkbox
(`enabled`), three sliders (`lin_vel_x`/`lin_vel_y`/`ang_vel_z`), one button (`zero`) —
declarative data the React app renders and drives via `setValue`/`triggerButton`.
`OnnxCommand`'s generic override (compute always; if `enabled`, overwrite `command` with
slider values) is nothing velocity-specific.

**`SliderCommandConfig` gets an adjustable range, entirely client-side** (mjlab's
"Max `<label>`" meta-slider only rescales the value slider's drag range, symmetric around
zero; no engine state, no new verb):

```ts
export interface SliderRangeControl { min: number; max: number; step: number; default: number; }
export interface SliderCommandConfig {
  type: 'slider'; name: string; label: string;
  min: number; max: number; step: number; default: number;
  enabled_when?: string;
  adjustable_range?: SliderRangeControl;   // NEW
}
```

When present, the app renders a companion "Max `<label>`" slider and locally clamps the
main slider's displayed range to `[-value, value]` — client-side only, never sent via
`setValue`. Assumes symmetry around zero (matches the three velocity axes); asymmetric
ranges are a follow-up. `get_env_idx()` collapses trivially at N=1.

### 3b. `LiftingCommandCfg` → `OnnxCommand` config (retires `LiftingCommand.ts`)

`tasks/manipulation/mdp/commands.py`. Second data instantiation of `OnnxCommand`, no
lifting-specific engine code:

- **State:** just `target_pos` `(3,)` at N=1 — this *is* the `command`. `episode_success`
  / `metrics` are training-only; drop them.
- **No continuous update:** `_update_command()` is `pass`. `next_state =
  where(resample_mask, resample(...), prev_state)`, no post-update.
- **No `ui`** (mjlab has no `create_gui` for it — autonomous only; don't add one).
- **`entity_write`:** `_resample_command` samples the cube's own spawn pose/velocity
  (`object_pose_range`, yaw-only orientation) and writes it via
  `write_root_link_pose_to_sim` / `write_root_link_velocity_to_sim` — separate from
  `target_pos`. `policy.json` declares the entity (`cube`) and fields; the graph outputs
  the values; the engine applies them via the new generic apply primitive.
  **Timing:** this write happens where Command sits — after `wasm.forward()`, so the new
  pose is visible only from the next frame (the accepted one-substep lag, ADR §8). Must
  **not** be "fixed" with a second `forward()`.
- **`difficulty`:** a training curriculum knob; bake the played checkpoint's mode
  (Lift-Cube-Yam uses `"dynamic"`) as a static trace-time choice — not a runtime toggle.
- Randomness goes through the `rand` input like everything else.

## 4. Carried forward from ADR review (execute, do not re-litigate)

- **Event config lives in `config.json`** (scene-scoped, `ConfigScene.events` via
  `mjswan/manifest`), not `policy.json`. Obs/Term/Command go in `policy.json`.
- **Reuse `OnnxModule`'s `is_init`/`carry` recurrent-state convention**
  (`core/policy/OnnxModule.ts`) for stateful Command/Event state, generalized to named
  fields — no second state-threading mechanism.
- **Sweep existing `Math.random()` sites** (`DslEvent.ts` `sampleUniform`,
  `TrackingCommand.ts` `sampleRangeValue`/`sampleInitialFrame`) into the
  orchestrator-owned seeded PRNG (ADR §2), or bit-for-bit replay can't hold.
- **Interval/startup Events are required, not deferrable.** `EventManager.ts` today has
  only `onReset()`. Velocity-Flat/Rough uses `mode="interval"` (`push_robot`,
  `interval_range_s=(1.0,3.0)`) and `mode="startup"` (`foot_friction`, `encoder_bias`,
  `base_com`); Lift-Cube-Yam uses `mode="startup"`. `startup` runs once at init (no
  timer); `interval` needs the countdown-timer trigger (scalar `time_left` per ADR §5,
  not tensors).

## 5. Acceptance criteria

- [ ] RNG spy/replay harness (§2b) in place and used for every term with internal
      randomness before any Command/Event parity claim is trusted. **(done for reset)**
- [ ] Every `examples/mjlab/*` task: N-step rollout matches the mjlab reference within
      tolerance on observations and termination flags.
- [ ] Velocity-Flat/Rough: `fell_over`/`out_of_terrain_bounds` termination tracing;
      `mode="startup"` + `mode="interval"` dispatch (native timer, §4); the
      `UniformVelocityCommand` ONNX body + declarative GUI override (§3a; GUI override
      checked manually, not in the automated parity check).
- [ ] Lift-Cube-Yam: `mode="startup"` dispatch; the `LiftingCommand` ONNX body (§3b),
      incl. the object pose/velocity reset side effect via the native delta mechanism.
- [ ] Tracking/Mimic tasks: physics/Obs/Term/Action/reset-Event parity holds;
      `TrackingCommand` **not** required to be ONNX-traced yet — confirm the native impl
      still runs unchanged.
- [ ] No term's Python source ships to the browser as executable text.
- [ ] `src/mjswan/dsl/` and `scripts/verify_dsl_migration.py` removed once the new
      harness supersedes their coverage.

## Review checkpoint — findings from the first real trace run

Findings from tracing Cartpole's observations and reset Events, recorded before
generalizing (per the sequencing above):

1. **N=1 is load-bearing for Event graphs.** `reset_joints_by_offset` does
   `joint_pos.view(len(env_ids), -1)`; `len(env_ids)` bakes the env count (=1) into the
   traced graph as a constant (TorchScript `TracerWarning`). Correct at N=1 (ADR §5); a
   future N>1 would need this addressed. Not a blocker — it confirms the N=1 decision is
   real, not cosmetic.
2. **Resolved indices bake in for free.** `torch.tensor(joint_ids)` and `[:, joint_ids]`
   become graph constants during tracing — exactly §2's "static indices baked as graph
   constants," no extra machinery.
3. **The dynamic-slot Event path is implemented but unexercised by Cartpole.** Reset
   reads only constants (`default_joint_pos/vel`, `soft_joint_pos_limits`) + `rand`. The
   first term to exercise a live-state read is `push_by_setting_velocity`, which reads
   `root_link_vel_w` — a field not yet in the tracer's dynamic-field set. → task (a).
4. **`entity_write` capture must be built for root pose/velocity.** `trace_event_term`
   currently captures only `write_joint_state_to_sim`. `push_by_setting_velocity` and
   `LiftingCommand` write via `write_root_link_velocity_to_sim` /
   `write_root_link_pose_to_sim`; the capture proxy and the write-target descriptor need
   those kinds (fields: root pose, root velocity). This is the §3/§3b `entity_write`
   mechanism starting on the Python side. → task (a).
5. **Reset parity is exact because we replay mjlab's own draws.** With the recorded draw
   fed as `rand`, `clamp(default + rand, limits)` reproduces mjlab bit-for-close
   (`max|Δ|=0`). Because mjlab's draws are within range, `clamp` is a no-op on these
   inputs, so a clamp-bounds bug would not be caught by this specific check. Meets the
   brief's "realistic-draw parity" bar; an out-of-range `rand` injection case could be
   added to also cover the clamp explicitly.

**Task (a) — done.** The tracer now generalizes over write kinds (`joint_state`,
`root_pose`, `root_velocity` via `_WRITE_FIELDS` + `_WriteCaptureMixin`), classifies
event reads into dynamic inputs vs baked constants like observations, and emits an
`entity_write` descriptor (`write_targets`). Validated on Go1-Velocity-Flat
(`scripts/onnx_parity_velocity_events.py`):

6. **`push_robot` (interval) parity clean — dynamic-slot + root-velocity `entity_write`
   both exercised for the first time.** `rand_dim=6`, no constants; it reads live
   `root_link_vel_w` as a graph *input* and writes `write_root_link_velocity_to_sim`;
   `max|Δ|=0` over 16 replayed draws. `reset_robot_joints` (12 joints, `rand_dim=24`)
   also clean.
7. **Play mode pops `push_robot`** (`go1/env_cfgs.py:242`), so the light play env has no
   interval event; the probe re-adds it to a play env (N=1) rather than building the
   thousands-env training cfg. The interval *timer* is TS-side (out of scope here); the
   Python harness validates the term-body math on both `resample_mask` states.
8. **`reset_base` / `randomize_terrain` fall back to native cleanly (next work).**
   `reset_root_state_uniform` reads `env.scene.env_origins` (a scene-level constant) and
   branches on `asset.is_fixed_base` (a non-tensor attribute); `randomize_terrain` reads
   the `env.scene.terrain` object. The tracer needs (i) scene-level constant capture and
   (ii) constant capture of control-flow scalar attrs before these trace. `root_pose`
   `entity_write` is implemented but first exercised once `reset_base` lands.

**Task (a) follow-up — done.** The event recorder/replayer was generalized from a
`(entity, field)` slot map to **tagged keys** (`("data", entity, field)` /
`("scene", attr)` / `("attr", entity, attr)`), so scene-level constants
(`env.scene.env_origins`) and control-flow scalars (`asset.is_fixed_base`) are captured
and baked, and non-tensor branch values keep tracing on the same path:

9. **`reset_base` (`reset_root_state_uniform`) now traces — `root_pose` `entity_write`
   exercised for the first time.** `rand_dim=12` (pose 6 + velocity 6), constants
   `default_root_state` + `scene:env_origins`; it branches on `is_fixed_base` (baked
   False → floating-base path) and writes both `write_root_link_pose_to_sim` and
   `write_root_link_velocity_to_sim`. `max|Δ|≈1.2e-7` over 16 replayed draws (float32
   rounding through `quat_from_euler_xyz`/`quat_mul`), within tolerance.
10. **`randomize_terrain` stays native** — it writes a *model field* (terrain), not a
    joint/root state, so it falls to the "wrote nothing traceable" fallback. Model-field
    writes are the `mode="startup"` domain-randomization mechanism (`geom_friction`,
    `encoder_bias`, `body_com_offset`); that is separate `entity_write`-adjacent work
    tracked under §4 (startup dispatch), not part of the value-graph tracer.
11. **A name-collision bug was caught and fixed:** the new event replay proxies initially
    reused the observation replay class names (`_ReplayScene`/`_ReplayEntity`/
    `_ReplayData`), shadowing them module-wide and breaking the *observation* path. Event
    proxies are now `_EvReplay*`. (Covered by the Cartpole pytest, which exercises both.)

**§3 Command tracer — started.** `trace_command_term` traces a class-based
`CommandTerm`'s `_resample_command` (gated by `resample_mask`) + `_update_command`
(always) as a pure function, promoting hidden state to explicit graph I/O
(`forward(prev_state…, resample_mask, rand) → (next_state…, entity_write?)`). State is
injected as inputs and read back as outputs (declared shape/dtype for `policy.json`);
randomness is replayed through `rand`; the resample gate is `where(mask, resampled,
prev)` with `prev` cloned before `_resample_command`'s in-place writes; reset unifies to
`resample_mask=True`.

12. **`LiftingCommand` traced — first stateful Command to ONNX (§3b).** On
    Lift-Cube-Yam (difficulty=`dynamic`): state `target_pos`, `rand_dim=7` (3 target + 3
    cube pos + 1 yaw, exactly as §2b predicted), the cube respawn `entity_write`
    (`root_pose` + `root_velocity`) reuses the proven root-write capture, and both
    `resample_mask` states are validated (True → resampled, False → unchanged).
    `max|Δ|≈6e-8` over 16 replayed draws. `_update_command` is `pass`; no dynamic
    runtime read, so the whole body traces as a pure function of `(prev_state, rand)`.

13. **`UniformVelocityCommand` is blocked — the first term to hit §9's escape hatch
    territory (§3a).** Two obstacles, both structural: (i) it draws with the **tensor
    method** `r.uniform_(*range)` rather than the `sample_uniform` *function*, so the
    RNG spy (which patches module globals) doesn't see it — spying `Tensor.uniform_`
    means monkeypatching the tensor type, which is global and intrusive; (ii)
    `_resample_command`/`_update_command` use **data-dependent control flow** —
    `if len(fwd_ids) > 0` (forward-only envs), the `init_velocity` block, and
    `.nonzero()`+masked index-assign in `_update_command` (heading/world/standing) —
    which `torch.onnx.export` bakes to the trace-time branch rather than evaluating at
    runtime. Neither is a threading problem the current tracer can paper over; it needs
    either a trace-friendly masked rewrite of the term (`torch.where` instead of
    `nonzero`+branch) or the §9 native-TS treatment. This is the honest checkpoint
    before generalizing further.

    *Confirmed empirically* (`scripts/onnx_probe_velocity_command.py`): the
    `sample_uniform` spy records `rand_dim=0` (the `r.uniform_()` draws are invisible
    to it), and `trace_command_term` then **fails loudly** with
    `UnsupportedOperatorError: Exporting the operator 'aten::uniform' to ONNX opset
    version 17 is not supported` — the §9 escape hatch working as intended (a clear,
    named failure at export, not a silent miscompile). The RNG operator is the first
    wall; the data-dependent branches sit behind it.

14. **`UniformVelocityCommand`'s path — resolved by an examples-side trace-friendly
    override, no upstream mjlab change and no native-TS engine work.** The task author
    supplies a numerically-equivalent override of `_resample_command`/`_update_command`
    that uses `sample_uniform` (spyable) instead of `r.uniform_` and `torch.where`
    (branch-free) instead of `.nonzero()`+`if len(...)>0`, and swaps it onto the command
    before compiling — exactly ADR 0003's "authors write mjlab-style Python terms". This
    is strictly better than the two options weighed earlier (upstream fork; native TS):
    one authoring surface, Cloud-safe by construction, no engine PR. Parity is
    graph-vs-override (trace faithfulness); the override's equivalence to mjlab's original
    is a separate review/distribution-test concern (bit-parity is impossible across
    different RNG structures anyway). *Validated* (`scripts/onnx_command_override_demo.py`,
    Go1-Velocity-Flat): the overridden `twist` command traces to ONNX and holds parity —
    `rand_dim=4`, `max|Δ|=0` over 16 draws, both `resample_mask` states. The tracer needed
    two ONNX-compat fixes that generalize to all bool state fields: gate bool fields via
    int64 (`Where` has no bool-branch kernel), and feed bool graph inputs as bool (not
    float32).

    **Remaining for *full* velocity: dynamic-slot Command support.** Heading tracking /
    world-frame rotation read runtime robot state (`self.robot.data.heading_w`), which the
    current Command tracer would bake as a constant. The fix is the same tagged-key
    proxy already proven for Events (§a), applied to `term.robot` / `term._env`: swap them
    to recording proxies in discovery and replay proxies in trace, threading dynamic reads
    as graph inputs. The demo omits heading tracking for exactly this reason.

15. **Dynamic-slot Command support — done; full velocity override (incl. heading
    tracking) traces.** The Command tracer now swaps `term.robot`/`term._env` to the
    Event tagged-key proxies (`_RecordCommand` in discovery, `_EvReplay*`/`_EventReplayEnv`
    in the traced module), so a command's time-varying runtime reads
    (`self.robot.data.heading_w`) thread as dynamic graph inputs while scene constants
    (`env_origins`) and control-flow scalars still bake. *Validated* (override demo v2,
    Go1-Velocity-Flat): a trace-friendly `twist` with heading tracking (steer `ang_vel_z`
    toward `heading_target` via `wrap_to_pi` + `torch.where`, reading `heading_w`) traces
    and holds parity — state `[vel_command_b, heading_target, is_heading_env,
    is_standing_env]`, `rand_dim=6`, `max|Δ|=0` over 16 draws. The `run_command_parity`
    `mask=False` check was corrected to compare against `_update_command`-only on prev
    (velocity applies standing/heading every frame, so "state unchanged" was wrong). The
    entity-swap refactor left Lift clean (`rand_dim=7`, `max|Δ|≈6e-8`). One fix the first
    run surfaced: the module must use the event replay entity (`_EvReplayEntity`, which
    carries `captures`), not the observation-path `_ReplayEntity`.

    With this, the §3a path is complete: `UniformVelocityCommand` is fully traceable via a
    trace-friendly examples-side override (no upstream mjlab change, no native TS). The
    override authored here covers lin/ang resample + heading + standing; a
    production override would also port world-frame/forward/init_velocity (all expressible
    with the same `sample_uniform` + `torch.where` vocabulary).

16. **`OnnxCommand` `policy.json` config output — done.** `mjswan.compile.serialize`
    turns a `CommandExport` into the single generic `OnnxCommand` config entry the runtime
    consumes (one handler interprets every command; no engine class per command). It
    carries everything the handler needs: `state_fields` (each with **shape + dtype**),
    `command_field`, `rand_dim`, `input_slots` (dynamic runtime reads), `write_targets`
    (entity_write), and optional `resampling_time_range`/`debug_vis`/`ui`. The `ui` block
    (checkbox/sliders/button, §3a) is authored task-side, not derived from the trace.
    `write_command_artifact` writes `command/<name>.onnx` beside it. Shipped with an
    authoritative JSON Schema (`COMMAND_JSON_SCHEMA`, for browser-side load-time validation
    per §6) and a dependency-free `validate_command_config`. Unit-tested pure-Python
    (`tests/test_onnx_command_config.py`, velocity + lifting shapes) and emitted end-to-end
    from the real velocity-override trace (`scripts/onnx_command_override_demo.py`).

    This reuses the existing `config.json`/`policy.json` contract (brief §4) — the entry
    replaces a command's DSL/`UiCommand` mapping. Wiring it into `PolicyConfig`'s live
    serialization (so `Builder.build()` emits it) is the remaining builder-integration step,
    deferred with the rest of the DSL→ONNX build-output switch (brief §2 deferral note).

17. **§4 TypeScript runtime — the four foundational pieces are built and unit-tested**
    (62 vitest tests green, eslint + tsc clean; all headless, no browser or ORT needed):

    - **`core/rng.ts`** — the orchestrator-owned seeded PRNG (ADR §2). A fully specified
      xoshiro128** (not a library's, so a dependency bump can never alter a recorded
      replay), with `getState`/`setState` so a session resumes bit-for-bit, and
      `randVector(n, ranges)` to fill a term's `rand` input. This is the sink that
      `Math.random()` call sites get swept into.
    - **`core/event/triggers.ts`** — native `interval`/`startup`/`reset` dispatch, the
      genuinely-new functionality (`EventManager` had only `onReset()`). Ports the mjlab
      *semantics* as scalars per ADR §5: `IntervalTrigger` (countdown, interval resampling,
      overshoot carried so the average rate doesn't drift, one firing per tick even for a
      long `dt`, per-episode vs global reset), `StartupTrigger` (fires once),
      `ResetTrigger` (`min_step_count_between_reset` gating).
    - **`core/event/entityWrite.ts`** — the apply primitive `DslEvent` never had (brief §3's
      correction): take a value the graph already computed and write it to mjData.
      Covers `joint_state` (qpos/qvel at resolved joint ids), `root_pose` (free-joint
      7-vector), `root_velocity` (6-vector); the `"<kind>__<field>"` output naming mirrors
      the Python tracer's `_WRITE_FIELDS`, so the tests double as the cross-language
      contract. Degrades (returns false) on a fixed-base model instead of throwing inside
      the step loop.
    - **`core/command/OnnxCommand.ts`** — the single generic command handler. Owns the
      scalar resample timer, threads state across frames (`prev_<field>` → `next_<field>`,
      seeded from the config's declared shape/dtype), draws `rand` from the seeded PRNG,
      applies the mjlab-parity UI override (**the autonomous computation is never skipped**;
      the UI overwrites per axis afterward, §3a), and hands `entity_write` outputs to the
      apply primitive. Velocity and Lifting are pure data instantiations — no per-command
      engine class, as §3 requires.

    **The sync/async boundary, resolved.** `CommandTerm.update()`/`getCommand()` are
    synchronous but ORT-Web inference is not. `update()` *kicks off* inference and returns;
    `getCommand()` serves the last completed value; a frame arriving while inference is in
    flight is **skipped, never queued**, so the command cannot build a backlog. A resample
    that comes due during an in-flight frame is latched and carried into the next admitted
    frame (tested). The resulting one-frame lag is the property already accepted in ADR §8.

    The ONNX session is injected behind a two-method `OnnxSession` interface, so all of the
    above is testable headless with a fake — no ORT, no browser, no WASM.

**Next:** wire these into the engine: `EventManager` gains mode-aware dispatch driven by the
triggers (calling a term's ONNX graph only on frames a trigger fires — brief §4's
"fusion reduces graph count, never call frequency"), `CommandManager` registers
`OnnxCommand` with an ORT-backed session built from app-supplied bytes (ADR 0004 §4 — the
engine never fetches), and the existing `Math.random()` sites in `DslEvent`/`TrackingCommand`
are swept into the seeded PRNG. A separate track: model-field-write startup-DR events
(`geom_friction`/`encoder_bias`/`body_com_offset`) — currently native-fallback, needed for
Velocity/Lift `mode="startup"` parity.
