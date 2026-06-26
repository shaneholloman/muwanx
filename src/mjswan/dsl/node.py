"""DAG node representation for the declarative MDP DSL (see ADR 0003).

A :class:`Node` is one operation in the composition graph.  A :class:`NodeRef`
wraps a Node and supports Python operator overloading so authors can write
``a + b``, ``a > c``, ``a[2]``, etc. to build the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class Node:
    """One operator in the composition DAG.

    Attributes:
        op: Primitive op name, must exist in the engine's primitive registry
            (e.g. ``"Add"``, ``"RootAngVelB"``, ``"Param"``).
        inputs: Upstream nodes consumed by this op.  Order matters (e.g.
            ``Sub.inputs[0]`` is the left-hand side).
        attrs: Static attributes that do not flow through the DAG
            (e.g. ``{"entity": "robot"}`` for an entity-bound source).
    """

    op: str
    inputs: list[Node] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


class NodeRef:
    """Thin wrapper around :class:`Node` providing Python operator overloads.

    The DSL functions return ``NodeRef`` instances (not bare ``Node``\\ s) so
    that ``a - b``, ``abs(x)``, ``x[2]`` and similar expressions can build
    further nodes naturally.
    """

    __slots__ = ("_node",)

    def __init__(self, node: Node) -> None:
        self._node = node

    @property
    def node(self) -> Node:
        return self._node

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------
    def __add__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Add", self, other)

    def __radd__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Add", other, self)

    def __sub__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Sub", self, other)

    def __rsub__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Sub", other, self)

    def __mul__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Mul", self, other)

    def __rmul__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Mul", other, self)

    def __neg__(self) -> NodeRef:
        return NodeRef(Node(op="Neg", inputs=[self._node]))

    def __abs__(self) -> NodeRef:
        return NodeRef(Node(op="Abs", inputs=[self._node]))

    # ------------------------------------------------------------------
    # Comparison operators (build bool-valued nodes)
    # ------------------------------------------------------------------
    def __gt__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Gt", self, other)

    def __lt__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Lt", self, other)

    def __ge__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Ge", self, other)

    def __le__(self, other: NodeRef | float) -> NodeRef:
        return _binary("Le", self, other)

    # ------------------------------------------------------------------
    # Boolean operators (build bool-valued nodes)
    # ------------------------------------------------------------------
    def __or__(self, other: NodeRef) -> NodeRef:
        return _binary("Or", self, other)

    def __ror__(self, other: NodeRef) -> NodeRef:
        return _binary("Or", other, self)

    def __and__(self, other: NodeRef) -> NodeRef:
        return _binary("And", self, other)

    def __rand__(self, other: NodeRef) -> NodeRef:
        return _binary("And", other, self)

    def __invert__(self) -> NodeRef:
        return NodeRef(Node(op="Not", inputs=[self._node]))

    # ------------------------------------------------------------------
    # Indexing (static integer only — see ADR 0003)
    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> NodeRef:
        if not isinstance(index, int):
            raise TypeError(
                "DSL indexing requires a static integer; got "
                f"{type(index).__name__}.  Dynamic indices must use an explicit "
                "Index primitive."
            )
        return NodeRef(Node(op="Index", inputs=[self._node], attrs={"i": index}))


def _binary(op: str, lhs: NodeRef | float, rhs: NodeRef | float) -> NodeRef:
    return NodeRef(Node(op=op, inputs=[_as_node(lhs), _as_node(rhs)]))


def _as_node(value: NodeRef | float | int) -> Node:
    """Coerce a Python scalar into a ``Const`` node, or unwrap a ``NodeRef``."""
    if isinstance(value, NodeRef):
        return value.node
    if isinstance(value, (int, float)):
        return Node(op="Const", attrs={"value": float(value)})
    raise TypeError(
        f"DSL operand must be a NodeRef or a Python scalar; got {type(value).__name__}"
    )
