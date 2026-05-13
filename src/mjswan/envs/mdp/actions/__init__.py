"""Action configuration classes for mjswan.

Mirrors ``mjlab.envs.mdp.actions`` so that the following import pattern
translates directly::

    # mjlab
    from mjlab.envs.mdp.actions import JointPositionActionCfg

    # mjswan (identical)
    from mjswan.envs.mdp.actions import JointPositionActionCfg
"""

from .actions import (
    ActionTermCfg,
    BaseActionCfg,
    JointEffortActionCfg,
    JointPositionActionCfg,
    JointVelocityActionCfg,
    MuscleActivationActionCfg,
    SiteEffortActionCfg,
    TendonEffortActionCfg,
    TendonLengthActionCfg,
    TendonVelocityActionCfg,
)

__all__ = [
    "ActionTermCfg",
    "BaseActionCfg",
    "JointPositionActionCfg",
    "JointVelocityActionCfg",
    "JointEffortActionCfg",
    "MuscleActivationActionCfg",
    "TendonLengthActionCfg",
    "TendonVelocityActionCfg",
    "TendonEffortActionCfg",
    "SiteEffortActionCfg",
]
