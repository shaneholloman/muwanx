# MuscleMimic session context (2026-05-14)

## Branch: `muscle`

## Entry points
- Python: `examples/mjlab/musclemimic/main.py`
- TS observations: `examples/mjlab/musclemimic/MimicObservations.ts`
- Run: `uv run python examples/mjlab/musclemimic/main.py`

---

## Current status
ONNX inference runs without errors (1152-dim obs vector matches model).
Control is **not proper** — muscle movements are wrong.

**The last fix (initial state from clip) has been written but NOT yet rebuilt.**
Run the server to trigger a fresh build, then test in browser.

---

## Root causes identified and fixed

### ✅ 1. `policy_num_actions` — DONE, dist rebuilt
Muscle policies have no joint transmission → `policy_joint_names = []`.
TS runtime was throwing "missing policy_joint_names" error.

Files changed:
- `src/mjswan/template/src/core/policy/types.ts` — added `policy_num_actions?: number`
- `src/mjswan/template/src/core/policy/PolicyRunner.ts` — added `getConfig()` method; `numActions = policy_num_actions ?? policyJointNames.length`
- `src/mjswan/template/src/core/engine/runtime.ts` — validation: pass if either `policy_num_actions` or `policy_joint_names` is present
- `src/mjswan/policy.py` — added `policy_num_actions: int | None = None` dataclass field
- `src/mjswan/builder.py` — serializes `policy_num_actions` in both code paths (with/without `config_path`)
- `examples/mjlab/musclemimic/main.py` — reads ONNX output `dim_value` and sets on handles

### ✅ 2. `MimicLookahead` (290 dims) — DONE, dist rebuilt
Without it: 862 dims → ONNX expected 1152. Diff = 290 = `mimic_lookahead`.

`mimic_lookahead` = 5 steps × 58 dims/step:
- 17 sites × 3 = 51 (future clip site_xpos − current root_pos)
- 3 = delta root pos (clip.qpos[future, :3] − current root_pos)
- 3 = future root vel (clip.qvel[future, :3])
- 1 = phase (future_frame / max(nFrames−1, 1))
- k=5, stride=20 frames between steps

Files changed:
- `examples/mjlab/musclemimic/MimicObservations.ts` — added `MimicLookahead` class + `protected get rootPos()` in `MimicClipObsBase`
- `examples/mjlab/musclemimic/main.py` — changed `mimic_lookahead` from unsupported → `MimicLookahead` with `defaults={k:5, stride:20, ctrl_dt:0.01, n_clip_sites:17}`

### ⚠️ 3. Initial state from clip — CODE WRITTEN, NOT YET REBUILT
Policy trained with RSI (random clip frame start). Model resets to keyframe (T-pose) → observations massively out of distribution → garbage actions.

Fix: set `initial_qpos` / `initial_qvel` from clip frame 0 in policy JSON. Runtime applies them after keyframe reset.

Files changed:
- `src/mjswan/template/src/core/engine/runtime.ts`:
  - Added `private initialQpos: number[] | null` and `private initialQvel: number[] | null`
  - `loadPolicyConfig`: `this.initialQpos = Array.isArray(config.initial_qpos) ? config.initial_qpos : null;` (before first `resetSimulationState`)
  - `resetSimulationState`: after keyframe reset, copies `initialQpos`/`initialQvel` into `mjData.qpos`/`mjData.qvel`
- `examples/mjlab/musclemimic/main.py`:
  - `np.load(clip_path)` → `clip_npz["qpos"][0].tolist()` / `clip_npz["qvel"][0].tolist()`
  - Sets `handle._config.initial_qpos` and `handle._config.initial_qvel` on all policy handles

---

## Key facts

### Model
- `myoMimicFullbody-v0`, nq=89, nv=88, nu=354 muscle actuators (na=354)
- Actuator names in model order **= `actuator_names` in policy JSON order** ✓ (verified)
- Keyframe 0 = T-pose (not a walking frame)

### Clip (NPZ)
- Cached at: `~/.cache/huggingface/hub/datasets--amathislab--musclemimic-retargeted/snapshots/.../MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz`
- Shape: `qpos(484,89)`, `qvel(484,88)`, `site_xpos(484,17,3)`, `site_names(<U20, 17)`
- Site order in clip: `['upper_body_mimic', 'head_mimic', ..., 'pelvis_mimic'(idx 8), ...]`
  vs `SITE_NAMES = list(FULLBODY_BODY2SITES_FOR_MIMIC.values())` (model order, starts with pelvis)
- Permutation handled in `MimicSiteTarget`/`MimicSiteErr` via `clipSiteNames` lookup ✓
- NPZ warns on `metadata` (|O) + `njnt`/`nbody`/`nsite` (<i8) — harmless

### Action application
```typescript
ctrl[ctrlAdr[i]] = 1 / (1 + Math.exp(-5 * (action[i] - 0.5)));  // sigmoid(5*(a-0.5))
```
MyoSuite canonical mapping. Verified actuator order matches. ✓

### `initial_qpos`/`initial_qvel` in builder
The `policy.py` dataclass already has these fields. The builder serializes them via `getattr(policy, "initial_qpos", None)`. They go into the policy JSON at the top level.

---

## All 10 TS observation classes (currently in dist)

| Class | Dims | Notes |
|-------|------|-------|
| MimicQpos | 89 | full `mjData.qpos` |
| MimicQvel | 88 | `mjData.qvel * ctrl_dt` (0.01) |
| MimicAct | 354 | `mjData.act` (muscle state) |
| MimicSitePos | 51 | current site_xpos by name |
| MimicSiteTarget | 51 | clip site_xpos at cur frame (permuted) |
| MimicSiteErr | 51 | target − current |
| MimicClipRefQpos | 89 | clip.qpos[cur_frame] |
| MimicClipRefQvel | 88 | clip.qvel[cur_frame] (no scaling) |
| MimicClipPhase | 1 | cur_frame / nFrames |
| MimicLookahead | 290 | 5 steps × 58 dims |

---

## Potential remaining issues (after initial-state fix)

1. **Async clip loading on frame 0**: Clip-based obs return zeros ~100ms until NPZ loads → first-frame jitter. Self-corrects.
2. **Phase formula**: `MimicClipPhase` uses `idx/nFrames`; Python uses `idx/max(T-1,1)`. Minor difference; `MimicLookahead` correctly uses `max(nFrames-1,1)`.
3. **RSI randomization**: Currently always resets to frame 0. True RSI picks a random frame. Low priority for demo.
4. **`ObservationGroupCfg.to_list()` skip-unsupported**: Already fixed — skips terms where `func.unsupported_reason is not None`.

---

## Files changed vs `main` (uncommitted)

```
src/mjswan/template/src/core/policy/types.ts
src/mjswan/template/src/core/policy/PolicyRunner.ts
src/mjswan/template/src/core/engine/runtime.ts
src/mjswan/managers/observation_manager.py
src/mjswan/policy.py
src/mjswan/builder.py
examples/mjlab/musclemimic/main.py
examples/mjlab/musclemimic/MimicObservations.ts
```
