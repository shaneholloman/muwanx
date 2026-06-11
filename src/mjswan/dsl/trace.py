"""Trace a DSL-style Python function into a serializable composition graph.

The build calls :func:`trace_observation` / :func:`trace_termination` with the
author's function and the ``params`` from the ``*TermCfg``.  The result is a
JSON-compatible dict matching the on-wire format described in ADR 0003.
"""

from __future__ import annotations

from typing import Any, Callable

from .env import SymbolicEnv
from .node import Node, NodeRef


def _topological_order(root: Node) -> list[Node]:
    """Return ``root`` and its inputs in post-order (deps before consumer)."""
    order: list[Node] = []
    visited: set[int] = set()

    def visit(node: Node) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        for parent in node.inputs:
            visit(parent)
        order.append(node)

    visit(root)
    return order


def _serialize_graph(root: Node) -> tuple[list[dict[str, Any]], str]:
    """Convert the DAG rooted at ``root`` into the ONNX-shaped node list."""
    ordered = _topological_order(root)
    name_by_id: dict[int, str] = {}
    nodes: list[dict[str, Any]] = []
    for idx, node in enumerate(ordered):
        out_name = f"n{idx}"
        name_by_id[id(node)] = out_name
        entry: dict[str, Any] = {"op": node.op, "out": out_name}
        if node.inputs:
            entry["in"] = [name_by_id[id(parent)] for parent in node.inputs]
        if node.attrs:
            entry["attrs"] = dict(node.attrs)
        nodes.append(entry)
    return nodes, name_by_id[id(root)]


def _unwrap(result: Any) -> Node:
    """Convert the function's return value into a Node, with a clear error."""
    if isinstance(result, NodeRef):
        return result.node
    raise TypeError(
        "Traced DSL function must return a NodeRef (an expression built from "
        f"the symbolic env), got {type(result).__name__}.  Plain Python "
        "values can't reach the engine unchanged — wrap them with `param()` "
        "or arithmetic on a `NodeRef`."
    )


def _trace(func: Callable[..., Any], params: dict[str, Any]) -> Node:
    """Run *func* once with a symbolic env and return the root DAG node."""
    env = SymbolicEnv()
    result = func(env, **params)
    return _unwrap(result)


def _bake_obs_postproc(
    root: Node,
    scale: float | tuple[float, ...] | list[float] | None,
    clip: tuple[float, float] | list[float] | None,
    history_steps: int | None,
) -> Node:
    """Append the mjlab obs pipeline (clip → scale → history) as graph nodes.

    Matches the mjlab ordering ``compute -> clip -> scale -> history`` so a
    declarative observation needs no special engine handling for these.
    """
    out = root
    if clip is not None:
        lo, hi = clip
        out = Node(op="Clip", inputs=[out], attrs={"min": float(lo), "max": float(hi)})
    if scale is not None:
        if isinstance(scale, (list, tuple)):
            factor = Node(op="ConstVec", attrs={"values": [float(v) for v in scale]})
        else:
            factor = Node(op="Const", attrs={"value": float(scale)})
        out = Node(op="Mul", inputs=[out, factor])
    if history_steps is not None and history_steps > 1:
        out = Node(op="History", inputs=[out], attrs={"steps": int(history_steps)})
    return out


def trace_observation(
    func: Callable[..., Any],
    params: dict[str, Any],
    *,
    scale: float | tuple[float, ...] | list[float] | None = None,
    clip: tuple[float, float] | list[float] | None = None,
    history_steps: int | None = None,
) -> dict[str, Any]:
    """Trace *func* into an observation-kind term envelope.

    ``scale`` / ``clip`` / ``history_steps`` are baked into the graph as
    trailing nodes (see :func:`_bake_obs_postproc`), so the engine interprets
    a single self-contained graph.
    """
    root = _bake_obs_postproc(_trace(func, params), scale, clip, history_steps)
    nodes, output = _serialize_graph(root)
    return {"kind": "observation", "nodes": nodes, "output": output}


def trace_termination(
    func: Callable[..., Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Trace *func* into a termination-kind term envelope."""
    nodes, output = _serialize_graph(_trace(func, params))
    return {"kind": "termination", "nodes": nodes, "output": output}


def trace_event(func: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
    """Trace a reset-event builder into an event-kind envelope.

    Unlike obs/termination, an event builder returns a list of
    :class:`~mjswan.dsl.event.Mutation` descriptors (side-effects applied on
    reset), serialized into ``{"kind": "event", "mutations": [...]}``.
    """
    from .event import Mutation

    env = SymbolicEnv()
    result = func(env, **params)
    if not isinstance(result, list) or not all(isinstance(m, Mutation) for m in result):
        raise TypeError(
            "Traced DSL event function must return a list of Mutation objects "
            f"(built via mjswan.dsl event helpers), got {type(result).__name__}."
        )
    return {"kind": "event", "mutations": [m.to_dict() for m in result]}
