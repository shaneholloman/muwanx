# Muscle action term aligned with MyoMuscleActivationActionCfg

mjswan's `MuscleActivationActionCfg` mirrors the contract of myosuite4's `MyoMuscleActivationActionCfg` (per-actuator-name resolution, canonical sigmoid mapping `σ(5(a − 0.5))` when normalising) and adopts the `normalize: bool` toggle from myosuite4's separate `MuscleActionTermCfg`. We do not adapt `MuscleActionTermCfg` directly: it is myosuite4's forward-looking modular API and is currently unused by any registered task (musclemimic, elbow, leg walk, myouser all use `MyoMuscleActivationActionCfg`).

## Considered options

- **Adapt both `MyoMuscleActivationActionCfg` and `MuscleActionTermCfg`.** Rejected: `MuscleActionTermCfg` exposes `entity_name` + `normalize` only (no `actuator_names`), so translating it to mjswan's `MuscleActivationActionCfg` requires a separate code path that enumerates all entity muscles at adapt time. No registered upstream task currently uses it, so we have nothing to test against.
- **Adapt `MuscleActionTermCfg` only and treat `MyoMuscleActivationActionCfg` as legacy.** Rejected: every myo* task on the mjlab backend (`myoElbowPose1D6MFixed-v0`, `myoLegWalk-v0`, `myoSarcLegWalk-v0`, musclemimic, myouser) instantiates `MyoMuscleActivationActionCfg`. Dropping support would block the actual musclemimic integration this work was created for.
- **Treat sigmoid as a hardcoded default and skip the `normalize` toggle.** Rejected: myosuite4's `MuscleActionTerm.process_actions` already supports `normalize=False → clip(actions, 0, 1)` for models whose actions are pre-normalised. Adopting the same toggle keeps mjswan ready for those models without a future breaking change.

## Consequences

- mjlab adapter maps only `MyoMuscleActivationActionCfg → MuscleActivationActionCfg`. Tasks built on `MuscleActionTermCfg` will not translate until myosuite4 wires it into a registered task; the second alias is then a small addition.
- `MuscleActivationActionCfg.normalize` defaults to `True`, matching both `MuscleActionTermCfg.normalize` and the implicit always-sigmoid behaviour of `MyoMuscleActivationActionCfg`. Existing mjswan demos and the musclemimic integration remain unchanged.
- `normalize=False` semantics are pinned to myosuite4 (`clip(scale*a + offset, 0, 1)`), not legacy myosuite (raw pass-through), because mjswan lacks a downstream "renormalise" stage and the clipped form is the safer default for arbitrary `ctrlrange`.
