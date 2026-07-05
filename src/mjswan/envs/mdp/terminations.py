"""Built-in termination function sentinels for mjswan.

Each object mirrors a function from ``mjlab.envs.mdp.terminations``.
In mjswan these are **not** called at runtime — they are sentinel
objects that carry metadata mapping to the TypeScript termination class
used by the browser-side ``TerminationManager``.

Usage (identical to mjlab)::

    from mjswan.envs.mdp import terminations as term_fns
    from mjswan.managers.termination_manager import TerminationTermCfg

    TerminationTermCfg(func=term_fns.time_out, time_out=True)
    TerminationTermCfg(func=term_fns.bad_orientation, params={"limit_angle": 1.0})
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TerminationBinding:
    """Binding from an mjlab termination name to its browser implementation.

    See ADR 0003.  Declarative terminations are plain traceable callables
    passed to ``TerminationTermCfg(func=...)``; a binding is the mjlab-name
    resolution layer carrying a ``ts_src`` escape hatch or ``unsupported_reason``.

    Attributes:
        ts_name: The TypeScript termination class name in the
            ``Terminations`` registry (e.g. ``"TimeOut"``).
        defaults: Default parameters merged into the JSON config entry.
        unsupported_reason: If set, this sentinel is accepted for API
            compatibility but raises ``NotImplementedError`` at build time.
        ts_src: Absolute path to a ``.ts`` file that exports the class
            named ``ts_name``. When set, the file is injected into the
            browser bundle at build time so the custom termination class is
            available to the browser-side ``TerminationManager``. Leave
            ``None`` for built-in classes already present in
            ``terminations.ts``.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    unsupported_reason: str | None = None
    ts_src: str | None = None


# ---------------------------------------------------------------------------
# Custom termination registry
# ---------------------------------------------------------------------------

_custom_registry: dict[str, TerminationBinding] = {}
"""Maps mjlab termination function names to user-supplied bindings.

Populated via :func:`register_termination`. The mjlab adapter checks this
registry as a fallback after the built-in sentinel lookup fails.
"""


def register_termination(mjlab_name: str, sentinel: TerminationBinding) -> None:
    """Register a custom termination binding for an mjlab termination.

    Call this before :meth:`~mjswan.Builder.build` so the adapter can
    resolve the function and the builder can inject any custom TypeScript
    source into the browser bundle.

    Args:
        mjlab_name: The mjlab termination function name
            (e.g. ``"out_of_terrain_bounds"``).
        sentinel: A :class:`TerminationBinding` describing the browser-side
            implementation. Set ``unsupported_reason`` to mark the
            termination as unsupported. Set ``ts_src`` to the absolute path
            of a ``.ts`` file that exports the class named by ``ts_name``.
    """
    _custom_registry[mjlab_name] = sentinel


# ---------------------------------------------------------------------------
# Episode timeout
# ---------------------------------------------------------------------------

_TIME_OUT_NEVER = 1e18


def time_out(env, *, max_episode_length: float | None = None, **_unused):
    """Terminate when the episode length exceeds its maximum.

    Declarative DSL form (see ADR 0003).  Uses the stateful ``StepCount``
    primitive that increments per step and clears on reset.

    Browser playback typically has no fixed max length, so the default is a
    sentinel large enough to be unreachable in practice; an mjlab task may
    inject an explicit value via ``params``.

    mjlab: ``env.episode_length_buf >= env.max_episode_length``
    """
    del env  # only the symbolic step_count is used
    from ...dsl import step_count

    limit = _TIME_OUT_NEVER if max_episode_length is None else max_episode_length
    return step_count() >= limit


# ---------------------------------------------------------------------------
# Orientation / height checks
# ---------------------------------------------------------------------------


def bad_orientation(env, *, limit_angle: float, entity_name: str = "robot", **_unused):
    """Terminate when the asset's orientation exceeds a limit angle.

    Declarative DSL form (see ADR 0003).  Equivalent to
    ``|acos(-projected_gravity_b[2])| > limit_angle``.

    Requires ``params={"limit_angle": <radians>}``.

    mjlab: ``torch.acos(-projected_gravity[:, 2]).abs() > limit_angle``
    """
    from ...dsl import acos

    pg = env.entity(entity_name).data.projected_gravity_b
    return abs(acos(-pg[2])) > limit_angle


def root_height_below_minimum(
    env, *, minimum_height: float, entity_name: str = "robot", **_unused
):
    """Terminate when the asset's root height is below a minimum.

    Declarative DSL form (see ADR 0003).  Equivalent to
    ``root_link_pos_w[2] < minimum_height``.

    Requires ``params={"minimum_height": <meters>}``.

    mjlab: ``asset.data.root_link_pos_w[:, 2] < minimum_height``
    """
    return env.entity(entity_name).data.root_link_pos_w[2] < minimum_height


# ---------------------------------------------------------------------------
# Terrain bounds
# ---------------------------------------------------------------------------


def out_of_terrain_bounds(
    env, *, limit_x: float, limit_y: float, entity_name: str = "robot", **_unused
):
    """Terminate when the robot leaves a fixed terrain footprint.

    Declarative DSL form (see ADR 0003).  Equivalent to
    ``|root_pos.x| > limit_x or |root_pos.y| > limit_y``.

    Requires ``params={"limit_x": <m>, "limit_y": <m>}`` — pre-computed at
    Python build time from the ``TerrainGeneratorCfg``.

    mjlab: ``tasks/velocity/mdp/terminations.out_of_terrain_bounds``
    """
    pos = env.entity(entity_name).data.root_link_pos_w
    return (abs(pos[0]) > limit_x) | (abs(pos[1]) > limit_y)


def terrain_edge_reached(
    env,
    *,
    half_x: float,
    half_y: float,
    threshold_fraction: float = 0.95,
    entity_name: str = "robot",
    **_unused,
):
    """Terminate when the robot displaces from its spawn beyond a sub-terrain edge.

    Declarative DSL form (see ADR 0003).  The spawn position is captured at
    episode start via the stateful ``SpawnCapture`` primitive; the first two
    steps are skipped (via ``StepCount``) to avoid stale-position triggers,
    matching the mjlab/engine behaviour.

    Requires ``params={"half_x": <m>, "half_y": <m>}``; ``threshold_fraction``
    defaults to 0.95.  Pre-computed at Python build time from the
    ``TerrainGeneratorCfg``.

    mjlab: ``tasks/velocity/mdp/terminations.terrain_edge_reached``
    """
    from ...dsl import spawn_capture, step_count

    pos = env.entity(entity_name).data.root_link_pos_w
    spawn = spawn_capture(pos)
    moved_x = abs(pos[0] - spawn[0]) > half_x * threshold_fraction
    moved_y = abs(pos[1] - spawn[1]) > half_y * threshold_fraction
    return (step_count() > 2) & (moved_x | moved_y)


# ---------------------------------------------------------------------------
# Motion-tracking deviation
# ---------------------------------------------------------------------------


def bad_anchor_pos_z_only(env, *, threshold: float, **_unused):
    """Terminate when the tracking-reference anchor z diverges from the robot's.

    Declarative DSL form (see ADR 0003).  Equivalent to
    ``|ref_anchor_z - current_anchor_z| > threshold``.

    Requires ``params={"threshold": <m>}``.
    """
    del env  # tracking sources read the browser command manager directly
    from ...dsl import tracking_anchor_pos, tracking_current_anchor_pos

    ref_z = tracking_anchor_pos()[2]
    cur_z = tracking_current_anchor_pos()[2]
    return abs(ref_z - cur_z) > threshold


def bad_anchor_ori(env, *, threshold: float, **_unused):
    """Terminate when the tracking-reference anchor orientation diverges.

    Declarative DSL form (see ADR 0003).  Compares the z component of gravity
    projected into the reference anchor frame versus the current anchor frame.

    Requires ``params={"threshold": <unitless>}``.
    """
    del env
    from ...dsl import (
        const_vec,
        quat_apply_inv,
        tracking_anchor_quat,
        tracking_current_anchor_quat,
    )

    gravity = const_vec([0.0, 0.0, -1.0])
    motion_gz = quat_apply_inv(tracking_anchor_quat(), gravity)[2]
    robot_gz = quat_apply_inv(tracking_current_anchor_quat(), gravity)[2]
    return abs(motion_gz - robot_gz) > threshold


def bad_motion_body_pos_z_only(
    env, *, threshold: float, body_names: list[str] | None = None, **_unused
):
    """Terminate when any tracked body z diverges from the reference.

    Declarative DSL form (see ADR 0003).  The engine's
    ``TrackingBodyPosZDeviationMax`` source reduces the per-body absolute z
    deviations to their maximum; the DSL compares that against ``threshold``.

    Requires ``params={"threshold": <m>}`` and optionally
    ``params={"body_names": [...]}`` to restrict the bodies checked (defaults
    to the motion command's full tracked-body list at runtime).
    """
    del env
    from ...dsl import tracking_body_pos_z_deviation_max

    return tracking_body_pos_z_deviation_max(body_names) > threshold


def base_ang_vel_exceed(
    env, *, threshold: float, entity_name: str = "robot", **_unused
):
    """Terminate when the base angular velocity exceeds ``threshold`` on any axis.

    Declarative DSL form (see ADR 0003) — the first MDP term migrated off
    the monolithic ``TerminationBinding`` sentinel + engine class pair.  Equivalent to
    ``any(|root_ang_vel_b| > threshold)``.

    Requires ``params={"threshold": <rad/s>}``.
    """
    from ...dsl import any_

    return any_(abs(env.entity(entity_name).data.root_ang_vel_b) > threshold)


# ---------------------------------------------------------------------------
# Safety / diagnostics (not supported in browser)
# ---------------------------------------------------------------------------

illegal_contact = TerminationBinding(
    ts_name="",
    unsupported_reason=(
        "illegal_contact is not supported in mjswan: contact force checks on "
        "specific bodies are not available in the browser runtime. "
        "This termination is a training-time safety check and is not needed "
        "for browser-side policy inference."
    ),
)
"""Terminate when a non-foot body makes illegal contact.

.. note::
    Not supported in mjswan. Accepted for API compatibility so that mjlab
    configs can be imported without modification, but raises
    ``NotImplementedError`` at build time.
"""

nan_detection = TerminationBinding(
    ts_name="",
    unsupported_reason=(
        "nan_detection is not supported in mjswan: NaN/Inf detection "
        "across the full physics state is not available in the browser runtime. "
        "The browser simulation will simply diverge visually if NaN occurs."
    ),
)
"""Terminate when NaN/Inf values appear in physics state.

.. note::
    Not supported in mjswan. Accepted for API compatibility so that mjlab
    configs can be imported without modification, but raises
    ``NotImplementedError`` at build time.
"""


__all__ = [
    "TerminationBinding",
    "register_termination",
    "time_out",
    "bad_orientation",
    "root_height_below_minimum",
    "illegal_contact",
    "nan_detection",
    "out_of_terrain_bounds",
    "terrain_edge_reached",
    "bad_anchor_pos_z_only",
    "bad_anchor_ori",
    "bad_motion_body_pos_z_only",
    "base_ang_vel_exceed",
]
