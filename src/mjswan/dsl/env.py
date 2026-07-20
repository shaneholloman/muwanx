"""Symbolic ``env`` exposed to traced DSL functions (see ADR 0003).

Authors write::

    def my_term(env, threshold):
        ang_vel = env.entity("robot").data.root_ang_vel_b
        return any_(abs(ang_vel) > threshold)

When the build invokes ``my_term`` with an instance of :class:`SymbolicEnv`,
each attribute / method access constructs a primitive op node instead of
returning a live tensor.  The complete DAG ends up in ``policy.json``.

The set of slot names exposed by :class:`EntityData` is the *input slot
registry* described in ADR 0003 — keep it in sync with the engine.
"""

from __future__ import annotations

from .node import Node, NodeRef

# ---------------------------------------------------------------------------
# Input slot registry — keep in sync with engine primitive registry.
# Each entry is a (slot_name, engine_op) pair: the attribute the author types
# and the primitive op name emitted into the DAG.
# ---------------------------------------------------------------------------

_ENTITY_DATA_SLOTS: dict[str, str] = {
    # Root state (base frame derivations).  The engine tracks a single root
    # angular velocity, so both mjlab spellings map to the same `RootAngVelB`.
    "root_link_lin_vel_b": "RootLinkLinVelB",
    "root_link_ang_vel_b": "RootAngVelB",
    "root_ang_vel_b": "RootAngVelB",
    "root_link_pos_w": "RootLinkPosW",
    "root_link_quat_w": "RootLinkQuatW",
    "projected_gravity_b": "ProjectedGravityB",
    # Joint state.
    "joint_pos": "JointPos",
    "joint_vel": "JointVel",
    "default_joint_pos": "DefaultJointPos",
}


class EntityData:
    """Symbolic accessor for ``Entity.data`` fields (see mjlab ``Entity.data``)."""

    __slots__ = ("_entity_name",)

    def __init__(self, entity_name: str) -> None:
        self._entity_name = entity_name

    def __getattr__(self, name: str) -> NodeRef:
        engine_op = _ENTITY_DATA_SLOTS.get(name)
        if engine_op is None:
            raise AttributeError(
                f"Unknown entity data slot '{name}'.  Known slots: "
                f"{sorted(_ENTITY_DATA_SLOTS)}.  If this is a new mjlab data "
                "field, add it to the engine primitive registry and to "
                "mjswan.dsl.env._ENTITY_DATA_SLOTS together."
            )
        return NodeRef(Node(op=engine_op, attrs={"entity": self._entity_name}))


class Entity:
    """Symbolic accessor for ``env.scene[name]`` (an entity)."""

    __slots__ = ("name", "data")

    def __init__(self, name: str) -> None:
        self.name = name
        self.data = EntityData(name)


class SymbolicEnv:
    """The ``env`` object passed to traced DSL functions.

    Mirrors the mjlab ``ManagerBasedRlEnv`` surface the term functions touch,
    but every accessor returns a :class:`NodeRef` so the function body builds
    a DAG instead of computing a tensor.
    """

    def entity(self, name: str) -> Entity:
        return Entity(name)
