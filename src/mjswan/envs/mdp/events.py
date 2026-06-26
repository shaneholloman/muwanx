"""Built-in event function sentinels for mjswan.

Each object mirrors a function from ``mjlab.envs.mdp.events``.
These are sentinel objects that carry metadata mapping to the TypeScript
event class used by the browser-side ``EventManager``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EventBinding:
    """Binding from an mjlab event name to its browser implementation.

    See ADR 0003.  Declarative reset events are plain builders returning
    ``list[Mutation]``; a binding is the mjlab-name resolution layer for
    events still backed by an engine class (``ts_src`` or a built-in name).

    Attributes:
        ts_name: The TypeScript event class name in the ``Events`` registry.
        defaults: Default parameters merged into the JSON config entry.
        ts_src: Absolute path to a ``.ts`` file that exports the class
            named ``ts_name``. When set, the file is injected into the
            browser bundle at build time. Leave ``None`` for built-in classes.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    ts_src: str | None = None


# Backwards-compatible alias (pre-ADR-0003 name).
EventFunc = EventBinding


_custom_registry: dict[str, EventBinding] = {}


def register_event_func(mjlab_name: str, sentinel: EventBinding) -> None:
    """Register a custom ``EventFunc`` sentinel for an mjlab event function."""
    _custom_registry[mjlab_name] = sentinel


def reset_root_state_uniform(env, *, pose_range=None, velocity_range=None, **_unused):
    """Reset the free-joint root with uniform random pose sampling.

    Declarative DSL form (see ADR 0003).  Adds independent uniform offsets to
    the root x/y/z and composes a uniform-random yaw onto the root quaternion.

    Requires ``params={"pose_range": {"x": [lo, hi], "y": ..., "yaw": ...}}``.

    mjlab: ``mjlab.envs.mdp.events.reset_root_state_uniform``
    """
    del env, velocity_range  # velocity randomization is not browser-relevant
    from ...dsl import add_freejoint_pos, compose_freejoint_yaw

    pose_range = pose_range or {}
    mutations = [
        add_freejoint_pos(
            x=tuple(pose_range.get("x", (0.0, 0.0))),
            y=tuple(pose_range.get("y", (0.0, 0.0))),
            z=tuple(pose_range.get("z", (0.0, 0.0))),
        )
    ]
    yaw = pose_range.get("yaw")
    if yaw is not None:
        mutations.append(compose_freejoint_yaw(yaw[0], yaw[1]))
    return mutations


# ``reset_root_state_from_flat_patches`` is NOT a core built-in.  It reads
# terrain flat-patch data (``terrainData``) — an engine capability outside
# bounded linear algebra, and a mjswan browser enhancement rather than an mjlab
# term.  Tasks that want patch-based spawning provide it task-side via ``ts_src``
# and register it with ``register_event_func``.  See
# examples/mjlab/defaults/events and ADR 0003.


def randomize_terrain(env, **_unused):
    """No-op event preserved for mjlab config compatibility.

    Declarative DSL form (see ADR 0003): an empty mutation list.  mjlab's
    ``randomize_terrain`` resamples terrain state during reset; mjswan bakes
    the terrain mesh into the exported scene so there is nothing to resample.

    mjlab: ``mjlab.envs.mdp.events.randomize_terrain``
    """
    del env
    return []


def reset_joints_by_offset(
    env,
    *,
    position_range=(0.0, 0.0),
    velocity_range=(0.0, 0.0),
    entity_name=None,
    joint_names=None,
    joint_ids=None,
    **_unused,
):
    """Reset selected joints with uniform-random offsets, honouring joint limits.

    Declarative DSL form (see ADR 0003).

    Requires ``params={"position_range": [min, max], "velocity_range": [min, max]}``
    and optionally ``params={"entity_name": ..., "joint_names": [...], "joint_ids": [...]}``.

    mjlab: ``mjlab.envs.mdp.events.reset_joints_by_offset``
    """
    del env
    from ...dsl import add_joint_pos, add_joint_vel

    return [
        add_joint_pos(
            position_range[0],
            position_range[1],
            entity_name=entity_name,
            joint_names=joint_names,
            joint_ids=joint_ids,
            clip_to_limits=True,
        ),
        add_joint_vel(
            velocity_range[0],
            velocity_range[1],
            entity_name=entity_name,
            joint_names=joint_names,
            joint_ids=joint_ids,
        ),
    ]


__all__ = [
    "EventBinding",
    "EventFunc",
    "register_event_func",
    "reset_root_state_uniform",
    "randomize_terrain",
    "reset_joints_by_offset",
]
