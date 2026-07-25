"""Command-term configuration and registration.

Commands follow the mjlab model: each policy owns a dictionary of command
terms, and each term produces a vector consumed by observations.

The browser UI is represented as metadata on top of command terms. Manual
slider/button controls are therefore implemented as a built-in ``UiCommand``
term rather than as a separate command system.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

CommandType = Literal["slider", "button", "checkbox"]


@dataclass
class SliderConfig:
    """Configuration for a slider input exposed by a command term."""

    name: str
    label: str
    range: tuple[float, float] = (-1.0, 1.0)
    default: float = 0.0
    step: float = 0.01
    enabled_when: str | None = None
    """Optional input name in the same command group that enables this slider."""

    @property
    def min(self) -> float:
        return self.range[0]

    @property
    def max(self) -> float:
        return self.range[1]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "slider",
            "name": self.name,
            "label": self.label,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "default": self.default,
        }
        if self.enabled_when is not None:
            data["enabled_when"] = self.enabled_when
        return data


Slider = SliderConfig


@dataclass
class CheckboxConfig:
    """Configuration for a checkbox input exposed by a command term."""

    name: str
    label: str
    default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "checkbox",
            "name": self.name,
            "label": self.label,
            "default": self.default,
        }


Checkbox = CheckboxConfig


@dataclass
class ButtonConfig:
    """Configuration for a button input exposed by a command term."""

    name: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "button",
            "name": self.name,
            "label": self.label,
        }


Button = ButtonConfig

CommandInput: TypeAlias = SliderConfig | ButtonConfig | CheckboxConfig


@dataclass
class CommandUiConfig:
    """Optional UI metadata attached to a command term."""

    inputs: list[CommandInput] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": [inp.to_dict() for inp in self.inputs]}


@dataclass
class PendingCommandTrace:
    """An mjlab ``CommandTermCfg`` not yet traced to ONNX (ADR 0005 §3).

    Command tracing needs a live env (``mjlab_cfg.build(env)`` constructs the
    term instance ``trace_command_term`` runs) and an output directory for the
    ``.onnx`` bytes — neither is available at ``add_policy()`` time, so
    resolution is deferred to build time (:func:`mjswan._onnx_build.serialize_command`),
    mirroring observations/terminations/events.
    """

    mjlab_cfg: Any
    """The raw mjlab ``CommandTermCfg`` (has ``.build(env)``, ``.resampling_time_range``,
    ``.debug_vis``)."""

    state_fields: list[str]
    """Attribute names on the built term constituting its hidden state."""

    command_field: str
    """Which state field is the command value."""

    trace_override: Callable[[Any], None] | None = None
    """Optional hook mutating a freshly ``build()``-constructed term in place —
    e.g. rebinding ``_resample_command``/``_update_command`` to a trace-friendly
    implementation (ADR 0005 §3a's examples-side override pattern)."""

    ui: dict[str, Any] | None = None
    """Author-authored UI descriptor (checkbox/sliders/button, §3a). Not
    derivable from the trace. Already resolved to a concrete dict — a
    callable ``CommandBinding.ui`` is called with the mjlab cfg before this
    dataclass is constructed (see ``mjlab_adapter._adapt_command_cfg``)."""


@dataclass
class CommandTermConfig:
    """Serialized browser-side command-term configuration."""

    term_name: str
    params: dict[str, Any] = field(default_factory=dict)
    ui: CommandUiConfig | None = None
    pending_trace: PendingCommandTrace | None = None
    """When set, this term is not yet resolved — see :class:`PendingCommandTrace`."""

    def to_dict(self) -> dict[str, Any]:
        if self.pending_trace is not None:
            raise TypeError(
                f"CommandTermConfig({self.term_name!r}) is pending ONNX trace — "
                "use mjswan._onnx_build.serialize_command(name, cfg, env, out_dir) "
                "instead of to_dict() directly (the Builder does this automatically)."
            )
        data = {"name": self.term_name, **self.params}
        if self.ui is not None:
            data["ui"] = self.ui.to_dict()
        return data


@dataclass(frozen=True)
class CommandBinding:
    """Binding from an mjlab command-cfg class name to its browser command term.

    Three mutually-exclusive shapes (ADR 0005 §3):

    - **Native**: ``ts_name`` names a permanently-native TS class (e.g.
      ``TrackingCommand`` for ``MotionCommandCfg``) — ``serializer`` builds its
      params directly from the mjlab cfg, no tracing involved.
    - **ONNX-traced**: ``state_fields``/``command_field`` set — the mjlab cfg is
      built (``cfg.build(env)``) and traced to ONNX at build time (the shared
      ``OnnxCommand`` handler; ``ts_name``/``serializer`` are unused). Set
      ``trace_override`` when the term needs a trace-friendly rewrite first
      (ADR 0005 §3a) — e.g. tensor-method RNG or data-dependent control flow
      that ``torch.onnx.export`` cannot handle as-authored. ``ui`` may be a
      static dict or a ``(mjlab_cfg) -> dict`` callable when the descriptor
      depends on the task's own cfg (e.g. slider ranges from ``cfg.ranges`,
      which can differ per task) — resolved once the real cfg is known.
    - **``ts_src`` escape hatch / unsupported marker**: same as before ADR 0005.
    """

    ts_name: str = ""
    serializer: Callable[[Any], Mapping[str, Any]] | None = None
    ts_src: str | None = None
    state_fields: list[str] | None = None
    command_field: str | None = None
    trace_override: Callable[[Any], None] | None = None
    ui: dict[str, Any] | Callable[[Any], dict[str, Any]] | None = None

    @property
    def is_onnx_traced(self) -> bool:
        return self.state_fields is not None and self.command_field is not None


_custom_registry: dict[str, CommandBinding] = {}


def register_command(mjlab_name: str, spec: CommandBinding) -> None:
    """Register a custom command term adapter.

    ``mjlab_name`` should typically be the mjlab config class name, e.g.
    ``"LiftingCommandCfg"``.
    """

    _custom_registry[mjlab_name] = spec


def ui_command(inputs: list[CommandInput]) -> CommandTermConfig:
    """Create the built-in manual UI command term."""

    return CommandTermConfig(
        term_name="UiCommand",
        ui=CommandUiConfig(inputs=list(inputs)),
    )


def velocity_command(
    *,
    lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z: tuple[float, float] = (-1.0, 1.0),
    default_lin_vel_x: float = 0.5,
    default_lin_vel_y: float = 0.0,
    default_ang_vel_z: float = 0.0,
) -> CommandTermConfig:
    """Create a built-in UI command term for planar velocity control."""

    return ui_command(
        [
            SliderConfig(
                name="lin_vel_x",
                label="Forward Velocity",
                range=lin_vel_x,
                default=default_lin_vel_x,
                step=0.05,
            ),
            SliderConfig(
                name="lin_vel_y",
                label="Lateral Velocity",
                range=lin_vel_y,
                default=default_lin_vel_y,
                step=0.05,
            ),
            SliderConfig(
                name="ang_vel_z",
                label="Yaw Rate",
                range=ang_vel_z,
                default=default_ang_vel_z,
                step=0.05,
            ),
        ]
    )


def _serialize_motion_command(cfg: Any) -> dict[str, Any]:
    """Convert mjlab's ``MotionCommandCfg`` into browser tracking metadata."""
    data: dict[str, Any] = {
        "anchor_body_name": getattr(cfg, "anchor_body_name", ""),
        "body_names": list(getattr(cfg, "body_names", ()) or ()),
        "sampling_mode": getattr(cfg, "sampling_mode", "start"),
        "pose_range": {
            key: list(value)
            for key, value in (getattr(cfg, "pose_range", None) or {}).items()
        },
        "velocity_range": {
            key: list(value)
            for key, value in (getattr(cfg, "velocity_range", None) or {}).items()
        },
        "joint_position_range": list(getattr(cfg, "joint_position_range", (0.0, 0.0))),
    }
    entity_name = getattr(cfg, "entity_name", None)
    if entity_name:
        data["entity_name"] = entity_name
    return data


# Bridges mjlab's MotionCommandCfg (e.g. isaac_lab_tasks MotionCommandCfg) to the TrackingCommand term.
register_command(
    "MotionCommandCfg",
    CommandBinding(
        ts_name="TrackingCommand",
        serializer=_serialize_motion_command,
    ),
)


__all__ = [
    "Button",
    "ButtonConfig",
    "Checkbox",
    "CheckboxConfig",
    "CommandBinding",
    "CommandInput",
    "CommandTermConfig",
    "CommandType",
    "CommandUiConfig",
    "Slider",
    "SliderConfig",
    "_custom_registry",
    "register_command",
    "ui_command",
    "velocity_command",
]
