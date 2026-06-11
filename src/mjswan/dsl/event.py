"""Declarative reset-event mutations (see ADR 0003).

Events are *side-effects*, not value-returning expressions: on episode reset
they mutate ``mjData`` (joint positions/velocities, the free-joint root pose).
Rather than a value DAG, an event traces to a list of **mutation descriptors**
— pure data the engine's ``DslEvent`` interprets with no per-simulation code.

A reset-event binding is a callable ``func(env, **params) -> list[Mutation]``.
The symbolic ``env`` is accepted for signature parity with mjlab but the
mutations are built from the helpers below.

Example::

    def reset_joints_by_offset(env, *, position_range, velocity_range, **_):
        return [
            add_joint_pos(*position_range, clip_to_limits=True),
            add_joint_vel(*velocity_range),
        ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Supported uniform-sample distributions for the engine's DslEvent.
_DIST_UNIFORM = "uniform"
_DIST_UNIFORM_XYZ = "uniform_xyz"


@dataclass(frozen=True)
class Mutation:
    """A single declarative write applied on reset.

    Attributes:
        target: Write target — ``"joint_qpos"``, ``"joint_qvel"``,
            ``"freejoint_pos"`` or ``"freejoint_yaw"``.
        op: ``"add"`` (offset the current value), ``"set"`` (overwrite) or
            ``"compose"`` (quaternion-compose, only for ``freejoint_yaw``).
        sample: Sampling descriptor interpreted by the engine, e.g.
            ``{"dist": "uniform", "low": -0.1, "high": 0.1}``.
        select: Optional joint selection (``entity_name`` / ``joint_names`` /
            ``joint_ids``) for joint targets.
        clip_to_limits: Clamp the result to the joint's limits after writing.
    """

    target: str
    op: str
    sample: dict[str, Any]
    select: dict[str, Any] | None = None
    clip_to_limits: bool = False

    def to_dict(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "target": self.target,
            "op": self.op,
            "sample": self.sample,
        }
        if self.select:
            entry["select"] = {k: v for k, v in self.select.items() if v is not None}
        if self.clip_to_limits:
            entry["clip_to_limits"] = True
        return entry


def _joint_select(
    entity_name: str | None,
    joint_names: list[str] | None,
    joint_ids: list[int] | None,
) -> dict[str, Any]:
    return {
        "entity_name": entity_name,
        "joint_names": list(joint_names) if joint_names else None,
        "joint_ids": list(joint_ids) if joint_ids else None,
    }


def add_joint_pos(
    low: float,
    high: float,
    *,
    entity_name: str | None = None,
    joint_names: list[str] | None = None,
    joint_ids: list[int] | None = None,
    clip_to_limits: bool = True,
) -> Mutation:
    """Add an independent uniform offset to each selected joint's qpos."""
    return Mutation(
        target="joint_qpos",
        op="add",
        sample={"dist": _DIST_UNIFORM, "low": float(low), "high": float(high)},
        select=_joint_select(entity_name, joint_names, joint_ids),
        clip_to_limits=clip_to_limits,
    )


def add_joint_vel(
    low: float,
    high: float,
    *,
    entity_name: str | None = None,
    joint_names: list[str] | None = None,
    joint_ids: list[int] | None = None,
) -> Mutation:
    """Add an independent uniform offset to each selected joint's qvel."""
    return Mutation(
        target="joint_qvel",
        op="add",
        sample={"dist": _DIST_UNIFORM, "low": float(low), "high": float(high)},
        select=_joint_select(entity_name, joint_names, joint_ids),
    )


def add_freejoint_pos(
    x: tuple[float, float] = (0.0, 0.0),
    y: tuple[float, float] = (0.0, 0.0),
    z: tuple[float, float] = (0.0, 0.0),
) -> Mutation:
    """Add an independent uniform offset to the free-joint root x/y/z."""
    return Mutation(
        target="freejoint_pos",
        op="add",
        sample={
            "dist": _DIST_UNIFORM_XYZ,
            "x": [float(x[0]), float(x[1])],
            "y": [float(y[0]), float(y[1])],
            "z": [float(z[0]), float(z[1])],
        },
    )


def compose_freejoint_yaw(low: float, high: float) -> Mutation:
    """Compose a uniform-random yaw rotation onto the free-joint root quat."""
    return Mutation(
        target="freejoint_yaw",
        op="compose",
        sample={"dist": _DIST_UNIFORM, "low": float(low), "high": float(high)},
    )
