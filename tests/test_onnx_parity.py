"""ADR 0005 Phase-1 parity: exported ONNX term graphs vs the live mjlab env.

Layer: L3 (requires the ``examples`` extras — mjlab / torch / onnxruntime — and a
one-time warp CPU-kernel compile, so it is marked ``slow``/``mjlab`` and skipped
when those deps are absent).

Asserts that every value-returning observation term of mjlab's Cartpole task,
traced to ONNX by :mod:`mjswan.compile`, reproduces the live env output within
tolerance for every term over multiple steps — and that ``time_out`` is
classified as a native term rather than an ONNX graph.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "disable")

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("mjlab")

pytestmark = [pytest.mark.slow, pytest.mark.mjlab]


@pytest.fixture(scope="module")
def cartpole_report():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

    from mjswan.compile import run_parity

    cfg = cartpole_balance_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    return run_parity(env, obs_group="actor", n_steps=16, seed=0)


def test_all_terms_pass_parity(cartpole_report):
    assert cartpole_report.passed, "\n" + cartpole_report.summary()


def test_observation_terms_are_onnx(cartpole_report):
    onnx_terms = {t.name for t in cartpole_report.terms if t.representation == "onnx"}
    # The four Cartpole actor observation terms must all be ONNX graphs.
    assert {"cart_pos", "pole_angle", "cart_vel", "pole_vel"} <= onnx_terms


def test_time_out_is_native(cartpole_report):
    time_out = next(t for t in cartpole_report.terms if t.name == "time_out")
    assert time_out.representation == "native"


def test_every_onnx_obs_term_checked_every_step(cartpole_report):
    for t in cartpole_report.terms:
        if t.representation == "onnx" and t.kind == "observation":
            assert t.steps_checked == cartpole_report.n_steps
            assert t.max_abs_diff <= cartpole_report.atol


def test_reset_events_are_onnx_and_match(cartpole_report):
    events = {t.name: t for t in cartpole_report.terms if t.kind == "event"}
    assert {"reset_slider", "reset_hinge"} <= set(events)
    for name in ("reset_slider", "reset_hinge"):
        ev = events[name]
        assert ev.representation == "onnx"
        # reset_joints_by_offset draws one position + one velocity offset.
        assert ev.rand_dim == 2
        assert ev.steps_checked > 0
        assert ev.max_abs_diff <= cartpole_report.atol


# ---------------------------------------------------------------------------
# The Builder path serializes from the *task config*, while the parity harness
# above reads the env's own prepared managers. Those are two different sources
# for the same terms, and they diverged once already: an unresolved
# `SceneEntityCfg` made the Builder trace all of an entity's sites instead of
# the one the task names, silently widening `ee_to_cube` from 3 to 6. These
# tests pin the two paths together on the width mjlab itself computes.
# ---------------------------------------------------------------------------

_SIZE_TASKS = [
    "Mjlab-Cartpole-Balance",
    "Mjlab-Lift-Cube-Yam",
    "Mjlab-Velocity-Flat-Unitree-G1",
]


def _mjlab_observation_widths(env, group: str) -> dict[str, int]:
    """Per-term output width as mjlab's own resolved manager computes it."""
    om = env.observation_manager
    widths = {}
    for term_name in om.active_terms[group]:
        cfg = om.get_term_cfg(group, term_name)
        value = cfg.func(env, **cfg.params)
        widths[term_name] = int(value.reshape(1, -1).shape[-1])
    return widths


@pytest.mark.parametrize("task_id", _SIZE_TASKS)
def test_serialized_observation_widths_match_mjlab(task_id, tmp_path):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjswan._onnx_build import serialize_observation_group
    from mjswan.adapters.mjlab_adapter import adapt_observations

    cfg = load_env_cfg(task_id, play=True)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    expected = _mjlab_observation_widths(env, "actor")

    groups = adapt_observations({"policy": cfg.observations["actor"]})
    assert groups is not None
    entries = serialize_observation_group(groups["policy"], env, tmp_path, "policy")

    # These groups all fuse (ADR 0005 §4), so the per-term widths live in the
    # group's `layout` — the runtime still needs them to name each slice of the one
    # vector the graph emits.
    assert isinstance(entries, dict), "expected a fused group"
    actual = {term["name"]: term["size"] for term in entries["layout"]}
    assert actual == expected
    assert entries["size"] == sum(expected.values())
    # And the concatenated group width the policy network sees.
    assert sum(actual.values()) == sum(expected.values())
