"""MDP components for mjswan.

``actions`` mirrors ``mjlab.envs.mdp.actions``: Action stays a closed,
permanently-native set (ADR 0005 §7), so ``mjswan.envs.mdp.actions`` carries
real, directly-usable action-term configs, same import pattern as mjlab::

    from mjswan.envs.mdp.actions import JointPositionActionCfg

``observations`` / ``terminations`` / ``events`` are different: mjswan does
**not** reimplement mjlab's term functions (ADR 0005). Use mjlab's own
functions directly — ``from mjlab.envs.mdp import observations as obs_fns`` —
and pass them straight to ``ObservationTermCfg(func=obs_fns.base_lin_vel)``;
the build traces the real function to ONNX. mjswan's own
``observations``/``terminations``/``events`` submodules only carry the
``*Binding`` escape hatch — a hand-written TS class for a term tracing cannot
express — and the ``register_*`` override registry that binds one to an mjlab
name.
"""

from . import actions, observations, terminations
from .events import EventBinding
from .observations import ObservationBinding
from .terminations import TerminationBinding

# Umbrella type for the mjlab-name → hand-written-TS binding layer. Per-kind
# bindings (ObservationBinding / TerminationBinding / EventBinding, plus
# mjswan.command.CommandBinding) share the role.
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
