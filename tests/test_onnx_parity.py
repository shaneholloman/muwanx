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


def test_every_onnx_term_checked_every_step(cartpole_report):
    for t in cartpole_report.terms:
        if t.representation == "onnx":
            assert t.steps_checked == cartpole_report.n_steps
            assert t.max_abs_diff <= cartpole_report.atol
