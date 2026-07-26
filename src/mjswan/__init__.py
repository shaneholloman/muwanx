"""mjswan: Browser-based MuJoCo Playground

Interactive MuJoCo simulations with ONNX policies running entirely in the browser.
"""

__version__ = "0.8.2"

from .app import MjswanApp
from .builder import Builder
from .command import (
    Button,
    ButtonConfig,
    Checkbox,
    CheckboxConfig,
    CommandBinding,
    CommandInput,
    CommandTermConfig,
    CommandUiConfig,
    Slider,
    SliderConfig,
    register_command,
    ui_command,
    velocity_command,
)
from .envs.mdp import MdpBinding
from .envs.mdp.actions import (
    ActionTermCfg,
    JointEffortActionCfg,
    JointPositionActionCfg,
)
from .envs.mdp.events import EventBinding, register_event
from .envs.mdp.observations import ObservationBinding, register_observation
from .envs.mdp.terminations import TerminationBinding, register_termination
from .managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from .managers.termination_manager import TerminationTermCfg
from .motion import MotionConfig, MotionHandle
from .policy import PolicyConfig, PolicyHandle
from .project import ProjectConfig, ProjectHandle
from .scene import SceneConfig, SceneHandle
from .splat import SplatConfig, SplatHandle
from .trace_env import build_single_entity_trace_env
from .viewer import ViewerConfig

__all__ = [
    # Builder and App
    "Builder",
    "MjswanApp",
    # Handles
    "ProjectHandle",
    "SceneHandle",
    "SplatHandle",
    "PolicyHandle",
    "MotionHandle",
    # Configs
    "ProjectConfig",
    "SceneConfig",
    "ViewerConfig",
    "SplatConfig",
    "PolicyConfig",
    "MotionConfig",
    # MDP bindings (mjlab-name → browser impl; see ADR 0003).
    # Pre-0.8 register_*_func / register_command_term names remain as deprecated
    # function aliases via _compat (the *Binding class aliases were removed).
    "MdpBinding",
    "ObservationBinding",
    "register_observation",
    "EventBinding",
    "register_event",
    "TerminationBinding",
    "register_termination",
    # MDP config (mjlab-compatible)
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "ActionTermCfg",
    "JointPositionActionCfg",
    "JointEffortActionCfg",
    "TerminationTermCfg",
    # Commands
    "Slider",
    "SliderConfig",
    "Button",
    "ButtonConfig",
    "Checkbox",
    "CheckboxConfig",
    "CommandInput",
    "CommandTermConfig",
    "CommandBinding",
    "CommandUiConfig",
    "register_command",
    "ui_command",
    "velocity_command",
    # ONNX tracing (ADR 0005)
    "build_single_entity_trace_env",
]

# Deprecated pre-0.8 aliases (methods, classes, modules). Remove in 0.9.
from . import _compat  # noqa: E402, F401
