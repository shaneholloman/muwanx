"""Observation terms for the Gentle Humanoid tracking policy (ADR 0005).

Plain ``func(env, **params)`` bodies written against the live-env API — the same
one mjlab's own ``observations`` module uses — traced to ONNX at build time. This
replaces the declarative-DSL version of these terms: the ops are now ordinary
torch, and the two things the DSL had engine primitives for are state the browser
serves as graph inputs instead:

- ``TrackingRefField(field, step)`` → the tracking command's ``ref_root_pos_w`` /
  ``ref_root_quat_w`` / ``ref_joint_pos`` **windows**. Each is the reference
  trajectory sampled at every offset in the command's ``time_steps``, so one read
  covers the whole look-ahead and the term slices out the offset it wants (static
  indices, baked into the graph).
- ``TrackingIsReady()`` → the command's ``is_ready`` field, still multiplied in so a
  motion-coupled term is zeros until a clip is loaded.

Proprioceptive history is no longer built out of ``History`` + ``slice_``: the
sparse look-back offsets go on the term itself (``history_steps``) and the runtime
owns the ring buffer, so these bodies compute a single frame.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    normalize,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)

# Gravity direction the policy's projected-gravity terms are expressed against.
_DOWN = (0.0, 0.0, -1.0)

#: Default asset. Joint-space terms get `preserve_order=True`, since the clip and the
#: checkpoint are in `action_joint_names` order while mjlab's resolve sorts into the model's.
_ROBOT = SceneEntityCfg(name="robot")


def _rot6d_columns(quat: torch.Tensor) -> torch.Tensor:
    """6D rotation, column-major ``[r00, r10, r20, r01, r11, r21]``.

    The first two rotation-matrix columns, flattened column-by-column — the
    convention this policy was trained against. mjlab's own ``motion_anchor_ori_b``
    flattens the same six numbers row-major, hence the transpose.
    """
    mat = matrix_from_quat(quat)
    return mat[..., :2].transpose(-2, -1).flatten(-2, -1)


def _down(quat: torch.Tensor) -> torch.Tensor:
    """``(0, 0, -1)`` over ``quat``'s batch, as an add rather than an expand so the
    graph keeps the batch axis dynamic instead of baking the trace-time size."""
    return torch.zeros_like(quat[..., :3]) + torch.tensor(
        _DOWN, dtype=quat.dtype, device=quat.device
    )


# ---------------------------------------------------------------------------
# Motion-coupled terms — read the reference window, gated by `is_ready`.
# ---------------------------------------------------------------------------


def tracking(
    env,
    *,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = _ROBOT,
    **_,
):
    """Root-pose look-ahead: base-frame position deltas + current-relative rot6d.

    Layout: for every offset after the first, the reference displacement from the
    first offset expressed in that first frame (3 each); then, for every offset, the
    reference orientation relative to the robot's current one as rot6d (6 each).
    """
    command = env.command_manager.get_term(command_name)
    ref_pos = command.ref_root_pos_w
    ref_quat = command.ref_root_quat_w
    steps = ref_pos.shape[1]
    parts = [
        quat_apply_inverse(ref_quat[:, 0], ref_pos[:, i] - ref_pos[:, 0])
        for i in range(1, steps)
    ]
    quat_inv_cur = quat_inv(env.scene[asset_cfg.name].data.root_link_quat_w)
    parts += [
        _rot6d_columns(quat_mul(quat_inv_cur, ref_quat[:, i])) for i in range(steps)
    ]
    return torch.cat(parts, dim=-1) * command.is_ready


def target_joint_pos(
    env,
    *,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = _ROBOT,
    **_,
):
    """Reference joint targets over the window, then those minus the current pose."""
    command = env.command_manager.get_term(command_name)
    ref = command.ref_joint_pos
    current = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    steps = ref.shape[1]
    targets = [ref[:, i] for i in range(steps)]
    diffs = [ref[:, i] - current for i in range(steps)]
    return torch.cat(targets + diffs, dim=-1) * command.is_ready


def target_root_z(env, *, command_name: str = "motion", **_):
    """Reference root height (z) at each offset in the window."""
    command = env.command_manager.get_term(command_name)
    return command.ref_root_pos_w[:, :, 2] * command.is_ready


def target_projected_gravity(env, *, command_name: str = "motion", **_):
    """Gravity direction projected into each reference frame, L2-normalized."""
    command = env.command_manager.get_term(command_name)
    ref_quat = command.ref_root_quat_w
    parts = [
        normalize(quat_apply_inverse(ref_quat[:, i], _down(ref_quat[:, i])))
        for i in range(ref_quat.shape[1])
    ]
    return torch.cat(parts, dim=-1) * command.is_ready


# ---------------------------------------------------------------------------
# Proprioception — one frame each; `history_steps` is stacked by the runtime.
# ---------------------------------------------------------------------------


def root_ang_vel(env, *, asset_cfg: SceneEntityCfg = _ROBOT, **_):
    """Root angular velocity in the body frame."""
    return env.scene[asset_cfg.name].data.root_link_ang_vel_b


def projected_gravity(env, *, asset_cfg: SceneEntityCfg = _ROBOT, **_):
    """Projected gravity, L2-normalized (the policy's convention, not mjlab's raw)."""
    return normalize(env.scene[asset_cfg.name].data.projected_gravity_b)


def joint_pos(env, *, asset_cfg: SceneEntityCfg = _ROBOT, **_):
    """Absolute joint positions, in policy order."""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(env, *, asset_cfg: SceneEntityCfg = _ROBOT, **_):
    """Joint velocities, in policy order."""
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


# ---------------------------------------------------------------------------
# Deployment-shaped inputs.
# ---------------------------------------------------------------------------


def boot(env, **_):
    """Deployment boot flag — always disabled in browser replay."""
    del env
    return torch.zeros(1, 1)


def compliance(env, *, command_name: str = "compliance", **_):
    """UI compliance command → ``[enabled, enabled*force, enabled*force/0.05]``.

    A traced body rather than a native command forward, because it is arithmetic on
    the command rather than the command itself; the slot resolves against the
    browser-only ``UiCommand``.
    """
    command = env.command_manager.get_command(command_name)
    enabled = (command[:, 0:1] >= 0.5).to(command.dtype)
    force = enabled * command[:, 1:2]
    return torch.cat([enabled, force, force / 0.05], dim=-1)
