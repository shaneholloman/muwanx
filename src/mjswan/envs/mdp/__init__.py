"""MDP components for mjswan.

``actions`` mirrors ``mjlab.envs.mdp.actions``: Action stays a closed,
permanently-native set (ADR 0005 §7), so ``mjswan.envs.mdp.actions`` carries
real, directly-usable action-term configs, same import pattern as mjlab::

    from mjswan.envs.mdp.actions import JointPositionActionCfg

``observations`` / ``terminations`` / ``events`` are different: mjswan reimplements
none of mjlab's term functions. Pass mjlab's own straight to
``ObservationTermCfg(func=obs_fns.base_lin_vel)`` and the build traces them. These
submodules carry only the ``*Binding`` escape hatch and its ``register_*`` registry.
"""

from . import actions, observations, terminations
from .events import EventBinding
from .observations import ObservationBinding
from .terminations import TerminationBinding

# Umbrella type for the mjlab-name → hand-written-TS binding layer.
MdpBinding = ObservationBinding | TerminationBinding | EventBinding

__all__ = [
    "EventBinding",
    "MdpBinding",
    "ObservationBinding",
    "TerminationBinding",
    "actions",
    "observations",
    "terminations",
]
