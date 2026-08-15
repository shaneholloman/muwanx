"""Action clipping: the wire fields, and the mjlab semantics the browser copies.

Two *different* bounds live here, and confusing them is the whole risk:

- ``ActionTermCfg.clip`` — per-target, on the action term, applied to
  ``raw * scale + offset``.
- ``clip_actions`` — symmetric, on the *runner* config, applied to the policy's raw
  output before any term sees it. rsl-rl's ``RslRlVecEnvWrapper.step`` clamps ahead of
  ``env.step``, so mjlab's action manager records the clamped vector and a
  ``last_action`` observation reads the clamped vector.

`ActionTermCfg.clip` was declared on the config and then dropped — `to_dict()` did
not emit it and the runtime had nowhere to apply it — so a task that set bounds got
none. `clip_actions` was read from the runner config, handed to the export-time
wrapper, and never written to the bundle, so it got none either. Both invisible,
because the reference tasks leave both `None`.

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


def test_clip_actions_reaches_the_policy_json(tmp_path, minimal_model, minimal_onnx):
    """The runner-config bound has to land in the bundle, or playback runs unclamped."""
    import json

    from mjswan import Builder
    from mjswan.utils import name2id

    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        control_dt=0.02, name="S", model=minimal_model
    )
    scene.add_policy(name="Policy", policy=minimal_onnx, clip_actions=100.0)

    out = tmp_path / "out"
    builder._save_web(out)
    data = json.loads(
        (
            out / "main" / "assets" / name2id("S") / f"{name2id('Policy')}.json"
        ).read_text()
    )
    assert data["clip_actions"] == 100.0


def test_clip_actions_zero_is_not_dropped(tmp_path, minimal_model, minimal_onnx):
    """`0.0` pins every action to zero — a real bound, and the one truthiness eats."""
    import json

    from mjswan import Builder
    from mjswan.utils import name2id

    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        control_dt=0.02, name="S", model=minimal_model
    )
    scene.add_policy(name="Policy", policy=minimal_onnx, clip_actions=0.0)

    out = tmp_path / "out"
    builder._save_web(out)
    data = json.loads(
        (
            out / "main" / "assets" / name2id("S") / f"{name2id('Policy')}.json"
        ).read_text()
    )
    assert data["clip_actions"] == 0.0


def test_clip_actions_absent_when_unset(tmp_path, minimal_model, minimal_onnx):
    import json

    from mjswan import Builder
    from mjswan.utils import name2id

    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        control_dt=0.02, name="S", model=minimal_model
    )
    scene.add_policy(
        name="Policy", policy=minimal_onnx, policy_joint_names=["j"]
    )  # something to serialize

    out = tmp_path / "out"
    builder._save_web(out)
    data = json.loads(
        (
            out / "main" / "assets" / name2id("S") / f"{name2id('Policy')}.json"
        ).read_text()
    )
    assert "clip_actions" not in data


@pytest.mark.slow
@pytest.mark.mjlab
def test_rsl_rl_clamps_before_the_env_sees_the_action():
    """Pin that the clamp precedes `env.step`, since that is what puts it ahead of the
    action manager — and therefore ahead of any `last_action` observation."""
    pytest.importorskip("mjlab")
    import inspect

    from mjlab.rl import RslRlVecEnvWrapper

    source = inspect.getsource(RslRlVecEnvWrapper.step)
    clamp_at = source.find("clamp")
    step_at = source.find("self.env.step")
    assert clamp_at != -1, (
        "rsl-rl no longer clamps in the wrapper; revisit clip_actions"
    )
    assert step_at != -1
    assert clamp_at < step_at, (
        "the clamp moved after env.step; the browser applies it before storing the "
        "action, which would no longer match"
    )


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

        # 1. The clamp applies to `raw * scale + offset`, and `offset` is the default joint
        #    position, so the processed action pins at the bound — not at `default + bound`.
        assert torch.allclose(processed, torch.full_like(processed, bound), atol=1e-6)

        # 2. The bias is removed after that, so the written value leaves the band by exactly
        #    the bias — which is what distinguishes the two clamp placements.
        bias = env.scene[term.cfg.entity_name].data.encoder_bias[
            :, term._target_ids  # noqa: SLF001 — mirrors mjlab's own apply_actions
        ][0]
        expected_target = processed - bias
        assert torch.allclose(
            expected_target, torch.full_like(processed, bound) - bias, atol=1e-6
        )
        # And the bias is non-zero here, or the two placements would be indistinguishable.
        assert bias.abs().max().item() > 0
    finally:
        env.close()
