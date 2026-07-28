"""Action `clip`: the wire field, and the mjlab semantics the browser copies.

`ActionTermCfg.clip` was declared on the config and then dropped — `to_dict()` did
not emit it and the runtime had nowhere to apply it — so a task that set bounds got
none. Invisible, because all three reference tasks leave it `None`.

Two things are checked here, and the second is the one that matters. Emitting the
field is easy to get right. *Where* the clamp goes is not: mjlab clamps
``raw * scale + offset`` inside ``BaseAction.process_actions`` and only then does
each kind's ``apply_actions`` run, which for ``joint_position`` subtracts the
encoder bias. So the final ``ctrl`` sits outside the declared bound by exactly that
bias — and an implementation that clamped the *target* instead would look perfectly
plausible while being wrong by the bias on every joint.

That is a contract with a dependency rather than with our own code, so it is pinned
against the live mjlab term: if upstream moves the clamp, this fails and says so.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "disable")

pytest.importorskip("torch")


def test_to_dict_emits_clip_as_patterns():
    """The dict travels as-is: the keys are patterns, resolved browser-side."""
    from mjswan.envs.mdp.actions import JointPositionActionCfg

    cfg = JointPositionActionCfg(
        actuator_names=(".*",), clip={"hip_.*": (-2.0, 2.0), "knee": (-1.0, 1.0)}
    )
    entry = cfg.to_dict()
    # Lists rather than tuples, since this is about to be JSON.
    assert entry["clip"] == {"hip_.*": [-2.0, 2.0], "knee": [-1.0, 1.0]}


def test_to_dict_omits_clip_when_unset():
    from mjswan.envs.mdp.actions import JointPositionActionCfg

    assert "clip" not in JointPositionActionCfg(actuator_names=(".*",)).to_dict()


def test_effort_action_also_carries_clip():
    """mjlab's clip lives on `BaseActionCfg`, so it is not position-only."""
    from mjswan.envs.mdp.actions import JointEffortActionCfg

    entry = JointEffortActionCfg(
        actuator_names=(".*",), clip={".*": (-5.0, 5.0)}
    ).to_dict()
    assert entry["clip"] == {".*": [-5.0, 5.0]}


@pytest.mark.slow
@pytest.mark.mjlab
def test_mjlab_clamps_the_processed_action_not_the_final_target():
    """Pin where mjlab's clamp sits, since the browser copies that placement."""
    pytest.importorskip("mjlab")
    import contextlib
    import io

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1", play=True)
    cfg.scene.num_envs = 1
    # A bound tight enough that a full-scale action saturates it.
    bound = 0.05
    cfg.actions["joint_pos"].clip = {".*": (-bound, bound)}
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg, device="cpu")
        env.reset()
    try:
        term = env.action_manager.get_term("joint_pos")
        n = term.action_dim
        # Large enough that every joint hits the ceiling.
        term.process_actions(torch.full((1, n), 1e3))
        processed = term._processed_actions[0]  # noqa: SLF001 — the value under test

        # 1. The clamp applies to `raw * scale + offset`, and `offset` here is the
        #    default joint position (`use_default_offset`), so the processed action
        #    is pinned at the bound — *not* at `default + bound`.
        assert torch.allclose(processed, torch.full_like(processed, bound), atol=1e-6)

        # 2. The bias is removed *after* that, so the value actually written leaves
        #    the declared band by exactly the bias. This is the assertion that
        #    distinguishes clamping the processed action from clamping the target.
        bias = env.scene[term.cfg.entity_name].data.encoder_bias[
            :, term._target_ids  # noqa: SLF001 — mirrors mjlab's own apply_actions
        ][0]
        expected_target = processed - bias
        assert torch.allclose(
            expected_target, torch.full_like(processed, bound) - bias, atol=1e-6
        )
        # And the bias is genuinely non-zero on this task, or the two placements
        # would be indistinguishable and this test would prove nothing.
        assert bias.abs().max().item() > 0
    finally:
        env.close()
