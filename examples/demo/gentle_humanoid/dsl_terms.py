"""Declarative DSL builders for the Gentle Humanoid tracking observations.

These reproduce the four motion-command-coupled observations that previously
required custom TypeScript (`GentleHumanoidObservations.ts`).  Each reads the
active motion command's reference trajectory through the engine's
`TrackingRefField` source primitive at an 11-point look-ahead window, so the
term is a pure composition graph (Cloud-safe) rather than author code.

Every term multiplies its result by ``tracking_is_ready()`` to reproduce the
bespoke ``if (!ready) return zeros`` short-circuit exactly.  See ADR 0003 and
issue #79.
"""

from __future__ import annotations

from mjswan.dsl import (
    command_value,
    concat,
    const_vec,
    history,
    joint_pos,
    joint_vel,
    normalize,
    prev_action,
    quat_apply_inv,
    quat_inv,
    quat_mul,
    quat_to_rot6d_columns,
    slice_,
    tracking_is_ready,
    tracking_ref_joint_pos,
    tracking_ref_root_pos,
    tracking_ref_root_quat,
)

# The look-ahead/history offsets the policy was trained with.  Passed as a
# static ``params["future_steps"]`` list and unrolled at trace time.
_DEFAULT_FUTURE_STEPS = [0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16]


def _steps(future_steps: list[int] | None) -> list[int]:
    return [int(s) for s in (future_steps or _DEFAULT_FUTURE_STEPS)]


def gentle_humanoid_tracking(env, *, future_steps=None, entity_name="robot", **_):
    """Root-pose look-ahead: base-frame position deltas + current-relative rot6d.

    Layout: for steps[1:], ``quat_apply_inv(refQuat0, refPos_i - refPos_0)`` (3
    each), then for every step ``rot6d_columns(quat_inv(rootQuat) * refQuat)``.
    """
    steps = _steps(future_steps)
    base_pos = tracking_ref_root_pos(steps[0])
    base_quat = tracking_ref_root_quat(steps[0])
    parts = [
        quat_apply_inv(base_quat, tracking_ref_root_pos(s) - base_pos)
        for s in steps[1:]
    ]
    q_cur_inv = quat_inv(env.entity(entity_name).data.root_link_quat_w)
    parts += [
        quat_to_rot6d_columns(quat_mul(q_cur_inv, tracking_ref_root_quat(s)))
        for s in steps
    ]
    return concat(parts) * tracking_is_ready()


def gentle_humanoid_target_joint_pos(
    env, *, future_steps=None, entity_name="robot", **_
):
    """Reference joint targets over the window, then those minus the current pose."""
    del env
    steps = _steps(future_steps)
    current = joint_pos(entity=entity_name)
    targets = [tracking_ref_joint_pos(s) for s in steps]
    diffs = [tracking_ref_joint_pos(s) - current for s in steps]
    return concat(targets + diffs) * tracking_is_ready()


def gentle_humanoid_target_root_z(env, *, future_steps=None, **_):
    """Reference root height (z) at each look-ahead step."""
    del env
    steps = _steps(future_steps)
    return concat([tracking_ref_root_pos(s)[2] for s in steps]) * tracking_is_ready()


def gentle_humanoid_target_projected_gravity(env, *, future_steps=None, **_):
    """Gravity direction projected into each reference frame, L2-normalized."""
    del env
    steps = _steps(future_steps)
    down = const_vec([0.0, 0.0, -1.0])
    parts = [normalize(quat_apply_inv(tracking_ref_root_quat(s), down)) for s in steps]
    return concat(parts) * tracking_is_ready()


# The proprioceptive history offsets the policy was trained with (sparse, not
# contiguous), so a plain History count won't do — we stack `max+1` frames and
# select the wanted offsets.
_DEFAULT_HISTORY_STEPS = [0, 1, 2, 3, 4, 8, 12, 16, 20]


def _sparse_history(base, history_steps, width: int):
    """Select frames at sparse look-back offsets from a per-step ``base``.

    ``History(base, max+1)`` is a step-major stack ``[frame_t, frame_{t-1}, …]``;
    ``slice_`` picks each wanted frame (``o`` steps ago at ``o * width``).
    """
    offsets = [int(o) for o in (history_steps or _DEFAULT_HISTORY_STEPS)]
    full = history(base, max(offsets) + 1)
    return concat([slice_(full, o * width, width) for o in offsets])


def gentle_humanoid_boot(env, **_):
    """Deployment boot flag — always disabled in browser replay."""
    del env
    return const_vec([0.0])


def gentle_humanoid_compliance(env, *, command_name="compliance", **_):
    """UI compliance command → ``[enabled, enabled*force, enabled*force/0.05]``.

    Assumes the ``compliance`` UI command is present (it always is); the bespoke
    default-when-absent branch is intentionally dropped.
    """
    del env
    cmd = command_value(command_name)
    enabled = (cmd[0] >= 0.5) * 1.0  # bool comparison -> 0.0/1.0
    ef = enabled * cmd[1]
    return concat([enabled, ef, ef / 0.05])


def gentle_humanoid_root_ang_vel(env, *, history_steps=None, entity_name="robot", **_):
    """Root angular velocity, sampled at sparse look-back offsets."""
    base = env.entity(entity_name).data.root_ang_vel_b
    return _sparse_history(base, history_steps, 3)


def gentle_humanoid_projected_gravity(
    env, *, history_steps=None, entity_name="robot", **_
):
    """Projected gravity (L2-normalized), sampled at sparse look-back offsets."""
    down = const_vec([0.0, 0.0, -1.0])
    base = normalize(
        quat_apply_inv(env.entity(entity_name).data.root_link_quat_w, down)
    )
    return _sparse_history(base, history_steps, 3)


def gentle_humanoid_joint_pos(
    env, *, history_steps=None, num_joints, entity_name="robot", **_
):
    """Joint positions, sampled at sparse look-back offsets."""
    del env
    return _sparse_history(
        joint_pos(entity=entity_name), history_steps, int(num_joints)
    )


def gentle_humanoid_joint_vel(
    env, *, history_steps=None, num_joints, entity_name="robot", **_
):
    """Joint velocities, sampled at sparse look-back offsets."""
    del env
    return _sparse_history(
        joint_vel(entity=entity_name), history_steps, int(num_joints)
    )


def gentle_humanoid_prev_actions(env, *, history_steps=8, **_):
    """The last ``history_steps`` policy actions, stacked (step-major)."""
    del env
    return history(prev_action(), int(history_steps))


#: mjswan registration name -> builder, in the order they appear in the obs group.
BUILDERS = {
    "gentle_humanoid_tracking": gentle_humanoid_tracking,
    "gentle_humanoid_target_joint_pos": gentle_humanoid_target_joint_pos,
    "gentle_humanoid_target_root_z": gentle_humanoid_target_root_z,
    "gentle_humanoid_target_projected_gravity": gentle_humanoid_target_projected_gravity,
    "gentle_humanoid_boot": gentle_humanoid_boot,
    "gentle_humanoid_compliance": gentle_humanoid_compliance,
    "gentle_humanoid_root_ang_vel": gentle_humanoid_root_ang_vel,
    "gentle_humanoid_projected_gravity": gentle_humanoid_projected_gravity,
    "gentle_humanoid_joint_pos": gentle_humanoid_joint_pos,
    "gentle_humanoid_joint_vel": gentle_humanoid_joint_vel,
    "gentle_humanoid_prev_actions": gentle_humanoid_prev_actions,
}
