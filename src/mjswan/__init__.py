"""mjswan: Browser-based MuJoCo Playground

Interactive MuJoCo simulations with ONNX policies running entirely in the browser.
"""

__version__ = "0.7.1"

from .app import mjswanApp
from .builder import Builder
from .command import (
    Button,
    ButtonConfig,
    Checkbox,
    CheckboxConfig,
    CommandBinding,
    CommandInput,
    CommandTermConfig,
    CommandTermSpec,
    CommandUiConfig,
    Slider,
    SliderConfig,
    register_command_term,
    ui_command,
    velocity_command,
)
from .envs.mdp import MjlabMdpBinding
from .envs.mdp.actions import (
    ActionTermCfg,
    JointEffortActionCfg,
    JointPositionActionCfg,
)
from .envs.mdp.events import EventBinding, EventFunc, register_event_func
from .envs.mdp.observations import ObsBinding, ObsFunc, register_obs_func
from .envs.mdp.terminations import TermBinding, TermFunc, register_termination_func
from .managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from .managers.termination_manager import TerminationTermCfg
from .motion import MotionConfig, MotionHandle
from .policy import PolicyConfig, PolicyHandle
from .project import ProjectConfig, ProjectHandle
from .scene import SceneConfig, SceneHandle
from .splat import SplatConfig, SplatHandle
from .viewer_config import ViewerConfig

__all__ = [
    # Builder and App
    "Builder",
    "mjswanApp",
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
    # *Func names are kept as backwards-compatible aliases.
    "MjlabMdpBinding",
    "ObsBinding",
    "ObsFunc",
    "register_obs_func",
    "EventBinding",
    "EventFunc",
    "register_event_func",
    "TermBinding",
    "TermFunc",
    "register_termination_func",
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
    "CommandTermSpec",
    "CommandUiConfig",
    "register_command_term",
    "ui_command",
    "velocity_command",
]
