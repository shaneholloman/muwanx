"""Command-term parity: traced graph vs the live mjlab term (ADR 0005 §3).

Layer: L3 (needs the ``examples`` extras and a warp CPU-kernel compile, so
``slow``/``mjlab``).

`run_command_parity` existed and had been run by hand through
`scripts/onnx_parity_lift_command.py` and `onnx_probe_velocity_command.py` — the
same gap the task sweep in `test_onnx_parity.py` closed, still open for commands.
It matters most here: commands are the **most randomness-heavy** layer in the
system (`lift_height` draws 7 values per resample, the tracking RSI graph 41),
and ADR 0005 §2b asks for the RNG spy/replay harness on *every* term with
internal randomness before a Command parity claim is trusted.

The resolution goes through the same path the Builder takes — the registry's
`CommandBinding` supplies `state_fields`, `command_field` and any
`trace_override`, and `cfg.build(env)` constructs the term — rather than
hardcoding those the way the scripts do. So this checks the declaration that
actually ships, and a registration that drifts from its term fails here.

Run::

    MUJOCO_GL=disable pytest tests/test_onnx_command_parity.py
"""

from __future__ import annotations

import contextlib
import io
import os
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "disable")

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("mjlab")

pytestmark = [pytest.mark.slow, pytest.mark.mjlab]

# Every reference task with a traced command, and the command's name in its config.
#
# `Mjlab-Tracking-Flat-Unitree-G1`'s `motion` is deliberately absent: its
# `MotionCommandCfg` stays a *native* `TrackingCommand` (a clip lookup is a data
# lookup, not term math) and only its RSI jitter is traced, as a reset graph rather
# than a command graph — covered by `test_onnx_command_config.py`. The env also
# cannot be constructed offline; its motion clip is a W&B artifact.
COMMAND_TASKS = [
    # Stateful, with an `entity_write` side effect on the cube (§3b).
    pytest.param("Mjlab-Lift-Cube-Yam", "lift_height", id="lift-cube-yam"),
    # Heading tracking, threaded as a dynamic slot; needs the examples-side
    # trace-friendly override (§3a).
    pytest.param("Mjlab-Velocity-Flat-Unitree-G1", "twist", id="velocity-flat-g1"),
    pytest.param("Mjlab-Velocity-Flat-Unitree-Go1", "twist", id="velocity-flat-go1"),
]


@pytest.fixture(scope="module", autouse=True)
def _registrations() -> None:
    """The traced command bodies live author-side; load them before resolving.

    `examples/mjlab/defaults/commands` is what binds `UniformVelocityCommandCfg`
    and `LiftingCommandCfg` to their traced bodies. Without it the adapter raises
    for an unregistered class — which is how a missing import surfaces, and is the
    same footgun that cost `g1_spinkick` its RSI jitter.
    """
    pytest.importorskip("examples.mjlab.defaults.commands")


def _traced_command(task_id: str, command_name: str) -> tuple[Any, Any]:
    """The live term and its pending trace, resolved as the Builder resolves them."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjswan.adapters.mjlab_adapter import _adapt_command_cfg

    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = 1
    cfg.sim.nconmax = 200_000
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg, device="cpu")
        env.reset()

    adapted = _adapt_command_cfg(cfg.commands[command_name])
    pending = adapted.pending_trace
    assert pending is not None, (
        f"{task_id}'s {command_name!r} resolved to a native term "
        f"({adapted.term_name!r}); this file only covers traced commands"
    )
    # Mirrors `serialize_command`: build from the cfg, then let the registration's
    # override swap in its trace-friendly bodies before anything is traced.
    term = pending.mjlab_cfg.build(env)
    if pending.trace_override is not None:
        pending.trace_override(term)
    return env, (term, pending)


@pytest.fixture(scope="module")
def command_report(request):
    from mjswan.compile import run_command_parity

    task_id, command_name = request.param
    env, (term, pending) = _traced_command(task_id, command_name)
    try:
        yield run_command_parity(
            term,
            pending.state_fields,
            name=command_name,
            command_field=pending.command_field,
            n_draws=16,
        )
    finally:
        env.close()


_PARAMS = [pytest.param((p.values[0], p.values[1]), id=p.id) for p in COMMAND_TASKS]


@pytest.mark.parametrize("command_report", _PARAMS, indirect=True)
def test_traced_command_matches_the_live_term(command_report):
    assert command_report.passed, "\n" + str(command_report.note)


@pytest.mark.parametrize("command_report", _PARAMS, indirect=True)
def test_every_draw_was_actually_replayed(command_report):
    """A command that traced but was never *run* would pass the assertion above.

    `passed` is an AND over comparisons and an empty AND is True, so the count of
    replayed draws is what separates "agreed" from "never asked" — the same guard
    the task sweep carries.
    """
    assert command_report.steps_checked == 16, (
        f"{command_report.name} compared {command_report.steps_checked} of 16 draws"
    )
    assert command_report.max_abs_diff is not None
    assert command_report.max_abs_diff <= 1e-5


@pytest.mark.parametrize("command_report", _PARAMS, indirect=True)
def test_the_command_actually_draws_randomness(command_report):
    """§2b is about terms *with* randomness; a `rand_dim` of 0 would be vacuous.

    Both of these resample from a range, so a zero here means the tracer stopped
    seeing the draws — which would make the replay above compare nothing.
    """
    assert command_report.rand_dim and command_report.rand_dim > 0, (
        f"{command_report.name} traced with rand_dim={command_report.rand_dim}; "
        "the replay harness then has no randomness to replay"
    )
