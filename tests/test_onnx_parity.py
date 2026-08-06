"""ADR 0005 parity: exported ONNX term graphs vs the live mjlab env.

Layer: L3 (requires the ``examples`` extras — mjlab / torch / onnxruntime — and a
one-time warp CPU-kernel compile, so it is marked ``slow``/``mjlab`` and skipped
when those deps are absent).

Two scopes. **Cartpole** is asserted in detail — every observation term traced,
``time_out`` classified native, both reset Events replayed against recorded RNG draws
— because it is small enough for each claim to be specific. **Every other reference
task** goes through :func:`parity_sweep`, the same harness over a wider term set.

Run the whole thing with::

    MUJOCO_GL=disable pytest tests/test_onnx_parity.py -m mjlab

Note that the default CI job runs ``-m "not slow"`` *and* installs only the
``dev`` extra, so nothing in this file runs there — the sweep has its own
workflow (``.github/workflows/parity.yml``).
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
# The sweep: the same harness over every reference task, not just Cartpole, whose four
# scalar observations miss the bugs the wide tasks had — a frozen `height_scan` on
# Velocity-Rough, an unresolved `SceneEntityCfg` widening `ee_to_cube` on Lift.
# ---------------------------------------------------------------------------

# The reference tasks, with why each one earns its place in the sweep.
SWEEP_TASKS = [
    pytest.param("Mjlab-Cartpole-Swingup", id="cartpole-swingup"),
    # Command-state slots and site-indexed reads, where the `SceneEntityCfg` bug hid.
    pytest.param("Mjlab-Lift-Cube-Yam", id="lift-cube-yam"),
    # Builtin-sensor slots, `joint_pos_biased`, a traced termination.
    pytest.param("Mjlab-Velocity-Flat-Unitree-G1", id="velocity-flat-g1"),
    pytest.param("Mjlab-Velocity-Flat-Unitree-Go1", id="velocity-flat-go1"),
    # `height_scan`: a structured `RayCastSensor`, once baked as 187 constants.
    pytest.param("Mjlab-Velocity-Rough-Unitree-G1", id="velocity-rough-g1"),
    pytest.param("Mjlab-Velocity-Rough-Unitree-Go1", id="velocity-rough-go1"),
]

# Out of the sweep: the Tracking tasks need a W&B motion clip and cannot be built offline,
# and the Lift camera variants observe rendered images, which mjswan does not serve.


@pytest.fixture(scope="module")
def sweep_report(request):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjswan.compile import run_parity

    cfg = load_env_cfg(request.param, play=True)
    cfg.scene.num_envs = 1
    # The Rough terrain makes far more contacts than mjlab's one-env `nconmax` allows,
    # so without this they raise `nconmax overflow` at construction.
    cfg.sim.nconmax = 200_000
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        yield run_parity(env, obs_group="actor", n_steps=8, seed=0)
    finally:
        env.close()


@pytest.mark.parametrize("sweep_report", SWEEP_TASKS, indirect=True)
def test_every_term_matches_mjlab(sweep_report):
    assert sweep_report.passed, "\n" + sweep_report.summary()


@pytest.mark.parametrize("sweep_report", SWEEP_TASKS, indirect=True)
def test_no_term_is_silently_unchecked(sweep_report):
    """A term reported as a graph must have been *compared*, every step.

    Without this the suite above passes on a task whose terms all traced and none
    of which was ever run: `passed` is an AND over comparisons, and an empty AND
    is True. `steps_checked` is what distinguishes "agreed" from "never asked".
    """
    graphs = [
        t
        for t in sweep_report.terms
        if t.representation == "onnx" and t.kind == "observation"
    ]
    assert graphs, "no observation term traced to a graph — the task serialized empty"
    for term in graphs:
        assert term.steps_checked == sweep_report.n_steps, (
            f"{term.name} compared on {term.steps_checked} of "
            f"{sweep_report.n_steps} steps"
        )
        assert term.max_abs_diff <= sweep_report.atol


# ---------------------------------------------------------------------------
# The Builder serializes from the task config while the harness above reads the env's
# prepared managers — two sources for the same terms, which diverged once when an
# unresolved `SceneEntityCfg` widened `ee_to_cube` from 3 to 6. Pinned here to the width
# mjlab itself computes.
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

    # These groups all fuse, so the per-term widths live in the group's `layout`.
    assert isinstance(entries, dict), "expected a fused group"
    actual = {term["name"]: term["size"] for term in entries["layout"]}
    assert actual == expected
    assert entries["size"] == sum(expected.values())
    # And the concatenated group width the policy network sees.
    assert sum(actual.values()) == sum(expected.values())
