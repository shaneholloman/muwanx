"""Free-function constructors for primitive ops the DSL exposes.

These complement :class:`NodeRef`'s operator overloads for cases where
Python syntax can't express the op (``abs(x)`` is built-in but reads
better as ``abs_(x)`` when chained; ``any_(...)`` has no operator form).
"""

from __future__ import annotations

from .node import Node, NodeRef, _as_node


def param(name: str) -> NodeRef:
    """Reference a config-time parameter (``params={name: ...}``)."""
    return NodeRef(Node(op="Param", attrs={"name": name}))


def abs_(x: NodeRef) -> NodeRef:
    """Elementwise absolute value."""
    return NodeRef(Node(op="Abs", inputs=[_as_node(x)]))


def any_(x: NodeRef) -> NodeRef:
    """Reduce a bool-valued tensor to a single bool via logical OR."""
    return NodeRef(Node(op="Any", inputs=[_as_node(x)]))


def all_(x: NodeRef) -> NodeRef:
    """Reduce a bool-valued tensor to a single bool via logical AND."""
    return NodeRef(Node(op="All", inputs=[_as_node(x)]))


def gt(x: NodeRef, y: NodeRef | float) -> NodeRef:
    """Elementwise greater-than."""
    return NodeRef(Node(op="Gt", inputs=[_as_node(x), _as_node(y)]))


def sub(x: NodeRef, y: NodeRef | float) -> NodeRef:
    """Elementwise subtraction."""
    return NodeRef(Node(op="Sub", inputs=[_as_node(x), _as_node(y)]))


def mul(x: NodeRef, y: NodeRef | float) -> NodeRef:
    """Elementwise multiplication."""
    return NodeRef(Node(op="Mul", inputs=[_as_node(x), _as_node(y)]))


def div(x: NodeRef, y: NodeRef | float) -> NodeRef:
    """Elementwise division (a scalar divisor broadcasts over a vector)."""
    return NodeRef(Node(op="Div", inputs=[_as_node(x), _as_node(y)]))


def sqrt(x: NodeRef) -> NodeRef:
    """Elementwise square root."""
    return NodeRef(Node(op="Sqrt", inputs=[_as_node(x)]))


def sum_(x: NodeRef) -> NodeRef:
    """Reduce a vector to the scalar sum of its elements."""
    return NodeRef(Node(op="Sum", inputs=[_as_node(x)]))


def acos(x: NodeRef) -> NodeRef:
    """Elementwise arccosine; clamps its argument to ``[-1, 1]`` (matching
    ``torch.acos`` numerical behaviour for slightly-out-of-range inputs)."""
    return NodeRef(Node(op="Acos", inputs=[_as_node(x)]))


def step_count() -> NodeRef:
    """Reference the episode step counter (stateful primitive).

    Increments on each evaluation; resets to 0 at the start of each
    episode (the engine's ``DslTermination.reset`` clears the state).
    """
    return NodeRef(Node(op="StepCount"))


def const_vec(values: list[float]) -> NodeRef:
    """A constant vector literal (e.g. a gravity direction ``[0, 0, -1]``)."""
    return NodeRef(Node(op="ConstVec", attrs={"values": [float(v) for v in values]}))


def spawn_capture(x: NodeRef) -> NodeRef:
    """Capture ``x`` on the first evaluation after reset and hold it.

    Stateful primitive: returns the value sampled at episode start on every
    subsequent step.  Used by spawn-relative terms (e.g. terrain-edge checks).
    """
    return NodeRef(Node(op="SpawnCapture", inputs=[_as_node(x)]))


def quat_apply_inv(quat: NodeRef, vec: NodeRef) -> NodeRef:
    """Rotate ``vec`` by the inverse of ``quat`` (MuJoCo ``(w, x, y, z)``)."""
    return NodeRef(Node(op="QuatApplyInv", inputs=[_as_node(quat), _as_node(vec)]))


# ---------------------------------------------------------------------------
# Motion-tracking sources (read the browser TrackingCommand at runtime)
# ---------------------------------------------------------------------------


def tracking_anchor_pos() -> NodeRef:
    """Reference anchor position (world frame) from the motion command."""
    return NodeRef(Node(op="TrackingAnchorPos"))


def tracking_anchor_quat() -> NodeRef:
    """Reference anchor orientation (world frame) from the motion command."""
    return NodeRef(Node(op="TrackingAnchorQuat"))


def tracking_current_anchor_pos() -> NodeRef:
    """Current robot anchor-body position (world frame)."""
    return NodeRef(Node(op="TrackingCurrentAnchorPos"))


def tracking_current_anchor_quat() -> NodeRef:
    """Current robot anchor-body orientation (world frame)."""
    return NodeRef(Node(op="TrackingCurrentAnchorQuat"))


def tracking_body_pos_z_deviation_max(body_names: list[str] | None = None) -> NodeRef:
    """Max ``|ref_z - current_z|`` over tracked bodies (world frame).

    Reads the motion command's reference body positions and the current
    robot body positions, returning the largest absolute z deviation.  When
    ``body_names`` is ``None`` the engine iterates the command's full tracked
    body list at runtime; pass a list to restrict the check.
    """
    attrs: dict[str, object] = {}
    if body_names is not None:
        attrs["body_names"] = list(body_names)
    return NodeRef(Node(op="TrackingBodyPosZDeviationMax", attrs=attrs))


def tracking_ref_body_pos(body_name: str) -> NodeRef:
    """Reference position of a single tracked body (world frame)."""
    return NodeRef(Node(op="TrackingRefBodyPos", attrs={"body": body_name}))


def _tracking_ref_field(field: str, step: int) -> NodeRef:
    return NodeRef(
        Node(op="TrackingRefField", attrs={"field": field, "step": int(step)})
    )


def tracking_ref_root_pos(step: int = 0) -> NodeRef:
    """Reference root position at ``refIdx + step`` (clamped to the clip)."""
    return _tracking_ref_field("root_pos", step)


def tracking_ref_root_quat(step: int = 0) -> NodeRef:
    """Reference root orientation at ``refIdx + step`` (clamped to the clip)."""
    return _tracking_ref_field("root_quat", step)


def tracking_ref_joint_pos(step: int = 0) -> NodeRef:
    """Reference joint targets at ``refIdx + step`` (clamped to the clip)."""
    return _tracking_ref_field("joint_pos", step)


def tracking_is_ready() -> NodeRef:
    """``1.0`` when a motion reference is loaded and ready, else ``0.0``."""
    return NodeRef(Node(op="TrackingIsReady"))


# ---------------------------------------------------------------------------
# Joint / sensor / command / pose sources
# ---------------------------------------------------------------------------


def _joint_names_attr(joint_names: str | list[str] | None) -> dict[str, object]:
    # A literal list selects explicit joints; "all"/None reads the policy
    # joint vector (engine treats a missing list that way).
    if isinstance(joint_names, list):
        return {"joint_names": list(joint_names)}
    return {}


def joint_pos(
    joint_names: str | list[str] | None = None, *, entity: str = "robot"
) -> NodeRef:
    """Joint positions; a name list selects joints, ``"all"``/None reads all."""
    return NodeRef(
        Node(op="JointPos", attrs={"entity": entity, **_joint_names_attr(joint_names)})
    )


def default_joint_pos(
    joint_names: str | list[str] | None = None, *, entity: str = "robot"
) -> NodeRef:
    """Default joint positions matching :func:`joint_pos`'s selection."""
    return NodeRef(
        Node(
            op="DefaultJointPos",
            attrs={"entity": entity, **_joint_names_attr(joint_names)},
        )
    )


def joint_vel(
    joint_names: str | list[str] | None = None, *, entity: str = "robot"
) -> NodeRef:
    """Joint velocities; a name list selects joints, ``"all"``/None reads all."""
    return NodeRef(
        Node(op="JointVel", attrs={"entity": entity, **_joint_names_attr(joint_names)})
    )


def prev_action() -> NodeRef:
    """The most recent policy action (wrap with :func:`history` to stack)."""
    return NodeRef(Node(op="PrevAction"))


def command_value(name: str) -> NodeRef:
    """The current value of a named command term."""
    return NodeRef(Node(op="CommandValue", attrs={"command": name}))


def sensor(name: str) -> NodeRef:
    """Raw data from a named MuJoCo sensor."""
    return NodeRef(Node(op="Sensor", attrs={"sensor": name}))


def site_pos(name: str) -> NodeRef:
    """World-frame position of a named site (current robot state)."""
    return NodeRef(Node(op="SitePos", attrs={"site": name}))


def body_pos(name: str) -> NodeRef:
    """World-frame position of a named body (current robot state)."""
    return NodeRef(Node(op="BodyPos", attrs={"body": name}))


def body_quat(name: str) -> NodeRef:
    """World-frame orientation of a named body (current robot state)."""
    return NodeRef(Node(op="BodyQuat", attrs={"body": name}))


# ---------------------------------------------------------------------------
# Vector / quaternion / scalar transforms
# ---------------------------------------------------------------------------


def concat(parts: list[NodeRef]) -> NodeRef:
    """Concatenate vectors along the feature axis, in order."""
    return NodeRef(Node(op="Concat", inputs=[_as_node(p) for p in parts]))


def slice_(v: NodeRef, start: int, length: int) -> NodeRef:
    """Extract the contiguous sub-range ``[start, start + length)`` of a vector."""
    return NodeRef(
        Node(
            op="Slice",
            inputs=[_as_node(v)],
            attrs={"start": int(start), "len": int(length)},
        )
    )


def history(x: NodeRef, steps: int, *, interleaved: bool = False) -> NodeRef:
    """Stack the most recent ``steps`` frames of ``x`` (step-major by default).

    A stateful primitive: the interpreter holds the ring buffer per node and
    clears it on episode reset.  ``interleaved`` lays the stack out joint-major.
    """
    attrs: dict[str, object] = {"steps": int(steps)}
    if interleaved:
        attrs["interleaved"] = True
    return NodeRef(Node(op="History", inputs=[_as_node(x)], attrs=attrs))


def cos(x: NodeRef) -> NodeRef:
    """Elementwise cosine."""
    return NodeRef(Node(op="Cos", inputs=[_as_node(x)]))


def sin(x: NodeRef) -> NodeRef:
    """Elementwise sine."""
    return NodeRef(Node(op="Sin", inputs=[_as_node(x)]))


def quat_mul(a: NodeRef, b: NodeRef) -> NodeRef:
    """Hamilton product of two quaternions (MuJoCo ``(w, x, y, z)``)."""
    return NodeRef(Node(op="QuatMul", inputs=[_as_node(a), _as_node(b)]))


def quat_inv(q: NodeRef) -> NodeRef:
    """Quaternion inverse."""
    return NodeRef(Node(op="QuatInv", inputs=[_as_node(q)]))


def quat_to_rot6d(q: NodeRef) -> NodeRef:
    """6D rotation representation (first two rotation-matrix columns, row-major)."""
    return NodeRef(Node(op="QuatToRot6d", inputs=[_as_node(q)]))


def quat_to_rot6d_columns(q: NodeRef) -> NodeRef:
    """6D rotation, column-major ``[r00, r10, r20, r01, r11, r21]``.

    Same six numbers as :func:`quat_to_rot6d` but ordered column-by-column,
    for policies trained against that convention.  A reindex composition of
    :func:`quat_to_rot6d` — not a dedicated engine op.
    """
    r = quat_to_rot6d(q)
    return concat([r[0], r[2], r[4], r[1], r[3], r[5]])


def normalize(v: NodeRef) -> NodeRef:
    """L2-normalize a vector: ``v / sqrt(sum(v * v))``.

    A composition of :func:`sqrt` / :func:`sum_` / division — not a dedicated
    engine op.  No zero-guard; callers ensure a non-zero input (the
    projected-gravity use rotates a unit vector, so the norm is always ~1).
    """
    return v / sqrt(sum_(v * v))
