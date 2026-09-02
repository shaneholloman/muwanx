"""The MDP a policy runs against, as one unit.

mjlab keeps observations, actions, terminations, commands and events side by side on
``ManagerBasedRlEnvCfg``; a checkpoint is trained against all five at once. mjswan
had spread them over the policy (four of them) and the scene (events), so a scene
hosting several checkpoints of one task traced the same MDP once per checkpoint and
could not switch its events with the policy. :class:`MdpConfig` puts the five back
together (ADR 0006 §3).

Identity is by object: two policies handed the *same* ``MdpConfig`` share one MDP —
one ``mdp/<id>/`` directory in the build, one set of traced graphs — while two equal
but separate configs are two MDPs. The per-manager keyword arguments on
:meth:`~mjswan.scene.SceneHandle.add_policy` remain and construct an anonymous one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .command import CommandTermConfig
    from .envs.mdp.actions.actions import ActionTermCfg
    from .managers.event_manager import EventTermCfg
    from .managers.observation_manager import ObservationGroupCfg
    from .managers.termination_manager import TerminationTermCfg


@dataclass
class MdpConfig:
    """The five term sets a policy is run against, held as one unit.

    Each field means what the matching :meth:`~mjswan.scene.SceneHandle.add_policy`
    argument means: ``None`` is "take it from the scene's env config" (or from the scene's
    own events), ``{}`` is "this MDP genuinely has none". The first policy to use a config
    on a scene fills its ``None`` fields in and adapts mjlab types in place, so what the
    build writes is what the object then holds.

    Example:
        mdp = mjswan.MdpConfig(observations=..., actions=..., terminations=...)
        scene.add_policy(name="model_1000", policy=..., mdp=mdp)
        scene.add_policy(name="model_2000", policy=..., mdp=mdp)  # shares it
    """

    observations: dict[str, ObservationGroupCfg] | Mapping[str, Any] | Any | None = None
    """Observation groups, keyed by the name the policy's ONNX slot table uses. A
    single group, mjlab's whole ``env_cfg.observations`` dict, or a dict already keyed."""

    actions: Mapping[str, ActionTermCfg] | Mapping[str, Any] | None = None
    """Action terms, keyed by term name (e.g. ``"joint_pos"``)."""

    terminations: dict[str, TerminationTermCfg] | dict[str, Any] | None = None
    """Termination terms, keyed by term name (e.g. ``"time_out"``, ``"fallen"``)."""

    commands: dict[str, CommandTermConfig] | Mapping[str, Any] | None = None
    """Command terms, keyed by their policy-visible names."""

    events: dict[str, EventTermCfg] | Mapping[str, Any] | None = None
    """Event terms, keyed by name, in any of the four modes. ``None`` takes the scene's
    events — a task's own, for an :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`
    scene — so a policy that says nothing about events gets the ones it was trained with."""

    name: str | None = None
    """Optional name. Its ``name2id`` is the MDP's id — its ``mdp/<id>/`` directory in the
    build; an unnamed config is ``mdp_<n>``, numbered per scene in first-use order."""

    _adapted: bool = field(default=False, init=False, repr=False, compare=False)
    """Set once the first policy has filled and adapted this config on a scene."""


__all__ = ["MdpConfig"]
