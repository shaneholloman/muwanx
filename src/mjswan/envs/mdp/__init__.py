"""MDP components for mjswan.

Mirrors ``mjlab.envs.mdp``.  Re-exports modules so that the following
mjlab import patterns translate directly::

    # mjlab
    from mjlab.envs.mdp import observations as obs_fns
    from mjlab.envs.mdp import terminations as term_fns
    from mjlab.envs.mdp.actions import JointPositionActionCfg

    # mjswan (identical)
    from mjswan.envs.mdp import observations as obs_fns
    from mjswan.envs.mdp import terminations as term_fns
    from mjswan.envs.mdp.actions import JointPositionActionCfg
"""

from . import actions, observations, terminations
from .events import EventBinding
from .observations import ObsBinding
from .terminations import TermBinding

# Umbrella type for the mjlab-name → browser-implementation binding layer
# (see ADR 0003).  Per-kind bindings (ObsBinding / TermBinding / EventBinding,
# plus mjswan.command.CommandBinding) share this role: resolving an mjlab name
# to a declarative builder, a ts_src escape hatch, or an unsupported marker.
MjlabMdpBinding = ObsBinding | TermBinding | EventBinding

__all__ = [
    "EventBinding",
    "MjlabMdpBinding",
    "ObsBinding",
    "TermBinding",
    "actions",
    "observations",
    "terminations",
]
