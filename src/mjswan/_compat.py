"""Backward-compatible aliases for the pre-0.8 mjswan API.

DELETE THIS MODULE in 0.9. Every name here is deprecated; each emits a
``DeprecationWarning`` and forwards to its replacement. Imported for its side
effects by :mod:`mjswan` — it installs method aliases on the handle classes, a
class-name alias for the app, and module aliases in :data:`sys.modules` for the
renamed modules.
"""

from __future__ import annotations

import functools
import sys
import warnings

from .app import MjswanApp
from .policy import PolicyHandle
from .project import ProjectHandle
from .scene import SceneHandle


def _alias_method(owner: type, old_name: str, new_name: str) -> None:
    """Install ``old_name`` on ``owner`` forwarding to ``new_name`` (warns)."""
    new = getattr(owner, new_name)

    @functools.wraps(new)
    def shim(self, *args, **kwargs):
        warnings.warn(
            f"{owner.__name__}.{old_name}() is deprecated and will be removed "
            f"in 0.9; use {owner.__name__}.{new_name}().",
            DeprecationWarning,
            stacklevel=2,
        )
        return new(self, *args, **kwargs)

    shim.__name__ = old_name
    shim.__qualname__ = f"{owner.__name__}.{old_name}"
    setattr(owner, old_name, shim)


def _alias_function(module, old_name: str, new_name: str) -> None:
    """Install module-level ``old_name`` forwarding to ``new_name`` (warns).

    The shim is installed on its owning ``module`` and on the ``mjswan``
    package namespace — both import paths were public pre-0.8.
    """
    new = getattr(module, new_name)

    @functools.wraps(new)
    def shim(*args, **kwargs):
        warnings.warn(
            f"{old_name}() is deprecated and will be removed in 0.9; use {new_name}().",
            DeprecationWarning,
            stacklevel=2,
        )
        return new(*args, **kwargs)

    shim.__name__ = old_name
    shim.__qualname__ = old_name
    setattr(module, old_name, shim)
    setattr(sys.modules["mjswan"], old_name, shim)


# Simple method renames.
_alias_method(ProjectHandle, "add_mjlab_scene", "add_scene_mjlab")
_alias_method(SceneHandle, "add_policy_from_wandb", "add_policy_wandb")
_alias_method(SceneHandle, "set_viewer_config", "set_viewer")
_alias_method(SceneHandle, "add_splat_section", "enable_splat_section")


# Param rename: add_motion_from_wandb(wandb_run_path=) -> add_motion_wandb(run_path=).
def _add_motion_from_wandb(self, *, wandb_run_path=None, **kwargs):
    warnings.warn(
        "PolicyHandle.add_motion_from_wandb() is deprecated and will be removed "
        "in 0.9; use PolicyHandle.add_motion_wandb(run_path=...).",
        DeprecationWarning,
        stacklevel=2,
    )
    if wandb_run_path is not None:
        kwargs["run_path"] = wandb_run_path
    return self.add_motion_wandb(**kwargs)


PolicyHandle.add_motion_from_wandb = _add_motion_from_wandb


# Class-name alias: mjswanApp -> MjswanApp (on both the package and submodule).
setattr(sys.modules["mjswan"], "mjswanApp", MjswanApp)
setattr(sys.modules["mjswan.app"], "mjswanApp", MjswanApp)


# Module aliases for the renamed modules (old import paths keep working).
from . import viewer as _viewer  # noqa: E402
from . import wandb_io as _wandb_io  # noqa: E402

sys.modules["mjswan.viewer_config"] = _viewer
sys.modules["mjswan.wandb_utils"] = _wandb_io
# Also expose as package attributes so ``mjswan.wandb_utils`` attribute access
# (not just ``import mjswan.wandb_utils``) resolves to the renamed module.
setattr(sys.modules["mjswan"], "viewer_config", _viewer)
setattr(sys.modules["mjswan"], "wandb_utils", _wandb_io)


# Pre-0.8 register_* function renames. The warning shims are installed on both
# the owning module and the ``mjswan`` namespace (both paths were public).
from . import command as _command  # noqa: E402
from .envs.mdp import events as _events  # noqa: E402
from .envs.mdp import observations as _observations  # noqa: E402
from .envs.mdp import terminations as _terminations  # noqa: E402

_alias_function(_observations, "register_obs_func", "register_observation")
_alias_function(_terminations, "register_termination_func", "register_termination")
_alias_function(_events, "register_event_func", "register_event")
_alias_function(_command, "register_command_term", "register_command")


# Pre-0.8 MDP binding class aliases. The leaf modules carry only the canonical
# ``*Binding`` names; these legacy aliases are restored here on every public
# import path they had pre-0.8. Silent — a type alias can't warn on attribute
# access without a per-module ``__getattr__``, which would not stay contained
# to this module. Removed in 0.9.
from .envs import mdp as _mdp  # noqa: E402


def _alias_class(alias: str, target: type, *modules) -> None:
    """Install class alias ``alias`` -> ``target`` on each module + ``mjswan``."""
    for module in (*modules, sys.modules["mjswan"]):
        setattr(module, alias, target)


_alias_class("ObsBinding", _observations.ObservationBinding, _observations, _mdp)
_alias_class("ObsFunc", _observations.ObservationBinding, _observations)
_alias_class("TermBinding", _terminations.TerminationBinding, _terminations, _mdp)
_alias_class("TermFunc", _terminations.TerminationBinding, _terminations)
_alias_class("EventFunc", _events.EventBinding, _events)
_alias_class("MjlabMdpBinding", _mdp.MdpBinding, _mdp)
_alias_class("CommandTermSpec", _command.CommandBinding, _command)
