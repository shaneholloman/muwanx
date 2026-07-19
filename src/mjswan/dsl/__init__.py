"""Declarative MDP term DSL (see ADR 0003).

Authors write mjlab-style functions taking a symbolic ``env``; calling the
function with an instance of :class:`SymbolicEnv` traces the body into a
DAG of primitive ops that the build serializes into ``policy.json`` and the
engine interprets at runtime.

Example::

    from mjswan.dsl import trace_observation, param

    def base_ang_vel_exceed(env, threshold):
        ang_vel = env.entity("robot").data.root_ang_vel_b
        return any_(abs_(ang_vel) > param("threshold"))

    graph = trace_termination(base_ang_vel_exceed, threshold=2.0)
    # graph -> {"kind": "termination", "inputs": [...], "nodes": [...], "output": "..."}
"""

from __future__ import annotations

from .env import Entity, EntityData, SymbolicEnv
from .event import (
    Mutation,
    add_freejoint_pos,
    add_joint_pos,
    add_joint_vel,
    compose_freejoint_yaw,
)
from .node import Node, NodeRef
from .ops import (
    abs_,
    acos,
    all_,
    any_,
    body_pos,
    body_quat,
    command_value,
    concat,
    const_vec,
    cos,
    default_joint_pos,
    div,
    gt,
    history,
    joint_pos,
    joint_vel,
    mul,
    normalize,
    param,
    prev_action,
    quat_apply_inv,
    quat_inv,
    quat_mul,
    quat_to_rot6d,
    quat_to_rot6d_columns,
    sensor,
    sin,
    site_pos,
    slice_,
    spawn_capture,
    sqrt,
    step_count,
    sub,
    sum_,
    tracking_anchor_pos,
    tracking_anchor_quat,
    tracking_body_pos_z_deviation_max,
    tracking_current_anchor_pos,
    tracking_current_anchor_quat,
    tracking_is_ready,
    tracking_ref_body_pos,
    tracking_ref_joint_pos,
    tracking_ref_root_pos,
    tracking_ref_root_quat,
)
from .trace import trace_event, trace_observation, trace_termination

__all__ = [
    "Entity",
    "EntityData",
    "Mutation",
    "Node",
    "NodeRef",
    "SymbolicEnv",
    "abs_",
    "add_freejoint_pos",
    "add_joint_pos",
    "add_joint_vel",
    "compose_freejoint_yaw",
    "acos",
    "all_",
    "any_",
    "body_pos",
    "body_quat",
    "command_value",
    "concat",
    "const_vec",
    "cos",
    "default_joint_pos",
    "div",
    "gt",
    "history",
    "joint_pos",
    "joint_vel",
    "mul",
    "normalize",
    "param",
    "prev_action",
    "quat_apply_inv",
    "quat_inv",
    "quat_mul",
    "quat_to_rot6d",
    "quat_to_rot6d_columns",
    "sensor",
    "sin",
    "site_pos",
    "slice_",
    "spawn_capture",
    "sqrt",
    "step_count",
    "sub",
    "sum_",
    "tracking_anchor_pos",
    "tracking_anchor_quat",
    "tracking_body_pos_z_deviation_max",
    "tracking_current_anchor_pos",
    "tracking_current_anchor_quat",
    "tracking_is_ready",
    "tracking_ref_body_pos",
    "tracking_ref_joint_pos",
    "tracking_ref_root_pos",
    "tracking_ref_root_quat",
    "trace_event",
    "trace_observation",
    "trace_termination",
]
