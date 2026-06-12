"""Built-in observation function sentinels for mjswan.

Each object mirrors a function from ``mjlab.envs.mdp.observations``.
In mjswan these are **not** called at runtime — they are sentinel
objects that carry metadata mapping to the TypeScript observation class
used by the browser-side ``PolicyRunner``.

Usage (identical to mjlab)::

    from mjswan.envs.mdp import observations as obs_fns
    from mjswan.managers.observation_manager import ObservationTermCfg

    ObservationTermCfg(func=obs_fns.base_lin_vel)
    ObservationTermCfg(func=obs_fns.joint_pos_rel, scale=0.5)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObsBinding:
    """Binding from an mjlab observation name to its browser implementation.

    See ADR 0003.  A binding is the *mjlab-name resolution* layer; authors of
    new declarative terms pass a traceable ``func=`` callable straight to
    ``ObservationTermCfg`` and bypass this entirely.  A binding carries either
    a ``ts_src`` custom-JS escape hatch or an ``unsupported_reason`` marker;
    declarative built-ins are plain callables, not bindings.

    Attributes:
        ts_name: The TypeScript observation class name in the
            ``Observations`` registry (e.g. ``"BaseLinearVelocity"``).
        defaults: Default parameters merged into the JSON config entry.
            These map mjlab semantics to the existing TS class API.
        unsupported_reason: If set, this sentinel is accepted for API
            compatibility but raises ``NotImplementedError`` at build time
            with this message.
        ts_src: Absolute path to a ``.ts`` file that exports the class
            named ``ts_name``.  When set, the file is injected into the
            browser bundle at build time so the custom observation class is
            available to the ``PolicyRunner``.  Leave ``None`` for built-in
            classes already present in ``observations.ts``.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    unsupported_reason: str | None = None
    ts_src: str | None = None


# Backwards-compatible alias (pre-ADR-0003 name). Kept so existing configs,
# examples and external users importing ``ObsFunc`` keep working.
ObsFunc = ObsBinding


# ---------------------------------------------------------------------------
# Custom observation registry
# ---------------------------------------------------------------------------

_custom_registry: dict[str, ObsBinding | Callable[..., Any]] = {}
"""Maps mjlab observation function names to a user-supplied binding.

Populated via :func:`register_obs_func`.  The mjlab adapter checks this
registry as a fallback after the built-in lookup fails.  An entry is either
an :class:`ObsBinding` (``ts_src`` escape hatch / unsupported marker) or a
**DSL builder callable** ``func(env, **params)`` for a task-specific
declarative term (ADR 0003).
"""


def register_obs_func(
    mjlab_name: str, sentinel: ObsBinding | Callable[..., Any]
) -> None:
    """Register a custom observation binding for an mjlab observation function.

    Call this before :meth:`~mjswan.Builder.build` so the adapter can resolve
    the function.  ``sentinel`` may be:

    - a **DSL builder callable** ``func(env, **params)`` — the build traces it
      into a composition graph (declarative, no ``ts_src``); this is how
      task-specific terms (e.g. a task's ``ee_to_object_distance``) stay out of
      the core library while remaining Cloud-safe.
    - an :class:`ObsBinding` with ``ts_src`` (custom-JS escape hatch) or
      ``unsupported_reason`` (accepted-but-unsupported marker).

    Args:
        mjlab_name: The mjlab observation function name
            (e.g. ``"ee_to_object_distance"``).
        sentinel: An :class:`ObsFunc` describing the browser-side
            implementation.  Set ``unsupported_reason`` to mark the
            observation as unsupported (silently skipped at build time).
            Set ``ts_src`` to the absolute path of a ``.ts`` file that
            exports the class named by ``ts_name``.

    Example — mark as unsupported::

        register_obs_func(
            "ee_to_object_distance",
            ObsFunc(ts_name="", unsupported_reason="not available in browser"),
        )

    Example — provide a custom TypeScript implementation::

        register_obs_func(
            "my_custom_obs",
            ObsFunc(ts_name="MyCustomObs", ts_src="/path/to/MyCustomObs.ts"),
        )
    """
    _custom_registry[mjlab_name] = sentinel


# ---------------------------------------------------------------------------
# Root state
# ---------------------------------------------------------------------------


def base_lin_vel(env, *, entity_name: str = "robot", **_unused):
    """Linear velocity of the robot base in the base frame.

    Declarative DSL form (see ADR 0003).  ``entity_name`` defaults to
    ``"robot"`` and is overridable so the mjlab adapter's ``asset_cfg``
    promotion flows through.  ``**_unused`` swallows other adapter-promoted
    kwargs (``world_frame``, etc.) that don't apply here.

    mjlab: ``asset.data.root_link_lin_vel_b``
    """
    return env.entity(entity_name).data.root_link_lin_vel_b


def base_ang_vel(env, *, entity_name: str = "robot", **_unused):
    """Angular velocity of the robot base in the base frame.

    Declarative DSL form (see ADR 0003).

    mjlab: ``asset.data.root_link_ang_vel_b``
    """
    return env.entity(entity_name).data.root_ang_vel_b


def projected_gravity(env, *, entity_name: str = "robot", **_unused):
    """Gravity vector projected into the base frame.

    Declarative DSL form (see ADR 0003).

    mjlab: ``asset.data.projected_gravity_b``
    """
    return env.entity(entity_name).data.projected_gravity_b


# ---------------------------------------------------------------------------
# Joint state
# ---------------------------------------------------------------------------


def joint_pos_rel(
    env,
    *,
    joint_names: str | list[str] = "all",
    entity_name: str = "robot",
    subtract_default: bool = True,
    default_joint_pos: list[float] | None = None,
    **_unused,
):
    """Joint positions relative to the default pose.

    Declarative DSL form (see ADR 0003).  A ``joint_names`` list selects
    specific joints; ``"all"`` reads the full policy joint vector.  When the
    scene-spec enrichment supplies explicit ``default_joint_pos`` values (e.g.
    a keyframe pose), those are subtracted as a constant; otherwise the engine
    reads the model defaults.

    mjlab: ``asset.data.joint_pos - asset.data.default_joint_pos``
    """
    del env
    from ...dsl import const_vec, joint_pos
    from ...dsl import default_joint_pos as default_joint_pos_op

    pos = joint_pos(joint_names, entity=entity_name)
    if subtract_default:
        if default_joint_pos is not None:
            pos = pos - const_vec(default_joint_pos)
        else:
            pos = pos - default_joint_pos_op(joint_names, entity=entity_name)
    return pos


def joint_vel_rel(
    env, *, joint_names: str | list[str] = "all", entity_name: str = "robot", **_unused
):
    """Joint velocities.

    Declarative DSL form (see ADR 0003).

    mjlab: ``asset.data.joint_vel`` (default velocities are zero)
    """
    del env
    from ...dsl import joint_vel

    return joint_vel(joint_names, entity=entity_name)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def last_action(env, **_unused):
    """The most recent action tensor.

    Declarative DSL form (see ADR 0003).  Stack past actions with the term's
    ``history_length``; pass ``params={"transpose": True}`` for the Isaac
    joint-major history layout (interleaved ``History``).

    mjlab: ``env.action_manager.action``
    """
    del env
    from ...dsl import prev_action

    return prev_action()


# Isaac-compatible alias (same as last_action; transpose via params).
previous_actions = last_action


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def generated_commands(env, *, command_name: str, **_unused):
    """Current command tensor from a named command term.

    Declarative DSL form (see ADR 0003).  Requires
    ``params={"command_name": "<name>"}``.

    mjlab: ``env.command_manager.get_command(command_name)``
    """
    del env
    from ...dsl import command_value

    return command_value(command_name)


# Isaac-named variants (projected_gravity_isaac, joint_positions_isaac,
# previous_actions) were redundant aliases of the canonical core terms
# (projected_gravity, joint_pos_rel, last_action) — the engine subclasses were
# empty.  The demo now references the canonical names directly.  The demo-only
# command obs (velocity_command_with_oscillators, impedance_command) live in
# the demo as DSL builders.  See examples/demo/main.py and ADR 0003.


# ``joint_pos_cos_sin`` is a single-task term (Cartpole pole angle) — it lives
# task-side now, composed from core primitives (cos/sin/concat/joint_pos).  See
# examples/mjlab/defaults/observations/__init__.py and ADR 0003.


# ---------------------------------------------------------------------------
# Motion tracking
# ---------------------------------------------------------------------------


def motion_anchor_pos_b(env, **_unused):
    """Reference anchor position relative to the current robot anchor frame.

    Declarative DSL form (see ADR 0003).  Equivalent to
    ``quat_apply_inv(current_anchor_quat, ref_anchor_pos - current_anchor_pos)``.
    """
    del env
    from ...dsl import (
        quat_apply_inv,
        tracking_anchor_pos,
        tracking_current_anchor_pos,
        tracking_current_anchor_quat,
    )

    diff = tracking_anchor_pos() - tracking_current_anchor_pos()
    return quat_apply_inv(tracking_current_anchor_quat(), diff)


def motion_anchor_ori_b(env, **_unused):
    """Reference anchor orientation relative to the current robot anchor frame.

    Declarative DSL form (see ADR 0003).  Returns a 6D rotation:
    ``rot6d(current_anchor_quat^-1 * ref_anchor_quat)``.
    """
    del env
    from ...dsl import (
        quat_inv,
        quat_mul,
        quat_to_rot6d,
        tracking_anchor_quat,
        tracking_current_anchor_quat,
    )

    rel = quat_mul(quat_inv(tracking_current_anchor_quat()), tracking_anchor_quat())
    return quat_to_rot6d(rel)


def robot_body_pos_b(env, *, body_names: list[str], **_unused):
    """Current robot body positions expressed in the current anchor frame.

    Declarative DSL form (see ADR 0003).  Statically unrolled over
    ``body_names`` (required at build time); concatenates
    ``quat_apply_inv(anchor_quat, body_pos - anchor_pos)`` per body.
    """
    del env
    from ...dsl import (
        body_pos,
        concat,
        quat_apply_inv,
        tracking_current_anchor_pos,
        tracking_current_anchor_quat,
    )

    anchor_pos = tracking_current_anchor_pos()
    anchor_quat = tracking_current_anchor_quat()
    parts = [
        quat_apply_inv(anchor_quat, body_pos(name) - anchor_pos) for name in body_names
    ]
    return concat(parts)


def robot_body_ori_b(env, *, body_names: list[str], **_unused):
    """Current robot body orientations expressed in the current anchor frame.

    Declarative DSL form (see ADR 0003).  Statically unrolled over
    ``body_names``; concatenates ``rot6d(anchor_quat^-1 * body_quat)`` per body.
    """
    del env
    from ...dsl import (
        body_quat,
        concat,
        quat_inv,
        quat_mul,
        quat_to_rot6d,
        tracking_current_anchor_quat,
    )

    anchor_inv = quat_inv(tracking_current_anchor_quat())
    parts = [
        quat_to_rot6d(quat_mul(anchor_inv, body_quat(name))) for name in body_names
    ]
    return concat(parts)


# Task-specific declarative observations (e.g. ee_to_object_distance,
# object_to_goal_distance for manipulation) are NOT defined here — they live in
# the task that uses them and register via ``register_obs_func`` with a DSL
# builder callable.  The core library carries only generic terms.  See
# examples/mjlab/defaults/observations/__init__.py and ADR 0003.


# ---------------------------------------------------------------------------
# Sensors (not supported in browser)
# ---------------------------------------------------------------------------


def builtin_sensor(env, *, sensor_name: str, **_unused):
    """Raw data from a named MuJoCo sensor.

    Declarative DSL form (see ADR 0003).  Requires
    ``params={"sensor_name": "<name>"}``; ``scale``/``clip`` apply via the
    term pipeline.
    """
    del env
    from ...dsl import sensor

    return sensor(sensor_name)


height_scan = ObsFunc(
    ts_name="",
    unsupported_reason=(
        "height_scan is not supported in mjswan: RayCastSensor is not "
        "available in the browser runtime."
    ),
)
"""Height scan from a RayCastSensor.

.. note::
    Not supported in mjswan. Accepted for API compatibility so that mjlab
    configs can be imported without modification, but raises
    ``NotImplementedError`` at build time.
"""


__all__ = [
    "ObsBinding",
    "ObsFunc",
    "register_obs_func",
    "_custom_registry",
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "previous_actions",
    "generated_commands",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "robot_body_pos_b",
    "robot_body_ori_b",
    "builtin_sensor",
    "height_scan",
]
