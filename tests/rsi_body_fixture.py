"""A reset-event body shaped like a real one, for the reset-graph tests.

Its own module on purpose: :class:`mjswan.compile.rng.DrawRecorder` swaps
``sample_uniform`` in the *body's module globals*, so a body that imports it
inside the function is invisible to the spy and traces with ``rand_dim=0``. Every
real term body (mjlab's own, and the examples' authored ones) imports at module
level; keeping this fixture honest about that means the test exercises the same
path a shipped body does.

Imported lazily by the test, so the suite still collects without mjlab.
"""

from __future__ import annotations

from typing import Any

from mjlab.utils.lab_api.math import sample_uniform


def rsi_joint_offset(env: Any, env_ids: Any, *, asset_cfg: Any, offset: float) -> None:
    """Read joint positions, add one drawn offset per joint, write them back."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos + sample_uniform(
        -offset,
        offset,
        asset.data.joint_pos.shape,
        device=asset.data.joint_pos.device,
    )
    asset.write_joint_state_to_sim(joint_pos, asset.data.joint_vel, env_ids=env_ids)
