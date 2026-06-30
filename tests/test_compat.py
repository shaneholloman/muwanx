"""Backward-compatibility coverage for the pre-0.8 aliases in ``mjswan._compat``.

Every assertion here targets a name scheduled for removal in 0.9. When that
cleanup happens, delete this file together with ``mjswan/_compat.py``.
"""

from __future__ import annotations

import warnings

import pytest

import mjswan
from mjswan.builder import Builder


def _policy(minimal_model, minimal_onnx):
    scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)
    return scene.add_policy(name="Policy", policy=minimal_onnx)


@pytest.mark.parametrize(
    ("owner_path", "old", "new"),
    [
        ("project.ProjectHandle", "add_mjlab_scene", "add_scene_mjlab"),
        ("scene.SceneHandle", "add_policy_from_wandb", "add_policy_wandb"),
        ("scene.SceneHandle", "set_viewer_config", "set_viewer"),
        ("scene.SceneHandle", "add_splat_section", "enable_splat_section"),
        ("policy.PolicyHandle", "add_motion_from_wandb", "add_motion_wandb"),
    ],
)
def test_deprecated_method_aliases_exist(owner_path, old, new):
    import importlib

    mod_name, cls_name = owner_path.split(".")
    owner = getattr(importlib.import_module(f"mjswan.{mod_name}"), cls_name)
    assert hasattr(owner, old)
    assert hasattr(owner, new)


def test_mjswanapp_alias_is_mjswanapp():
    assert mjswan.mjswanApp is mjswan.MjswanApp
    from mjswan.app import mjswanApp  # noqa: F401  (old import path still works)


def test_old_module_paths_resolve_to_renamed_modules():
    import mjswan.viewer_config
    import mjswan.wandb_utils

    import mjswan.viewer
    import mjswan.wandb_io

    assert mjswan.viewer_config is mjswan.viewer
    assert mjswan.wandb_utils is mjswan.wandb_io
    # attribute access on the package (not just import) also resolves
    assert mjswan.wandb_utils.resolve_wandb_run_path is (
        mjswan.wandb_io.resolve_wandb_run_path
    )


def test_enable_splat_section_alias_warns_and_works(minimal_model, minimal_onnx):
    scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)
    with pytest.warns(DeprecationWarning):
        result = scene.add_splat_section()
    assert result is scene
    assert scene._config.splat_section is True


def test_velocity_command_shortcuts_are_gone(minimal_model, minimal_onnx):
    policy = _policy(minimal_model, minimal_onnx)
    assert not hasattr(policy, "add_velocity_command")
    assert not hasattr(policy, "add_command_velocity")


def test_mdp_class_aliases_resolve_to_canonical():
    from mjswan.command import CommandBinding
    from mjswan.envs.mdp import MdpBinding
    from mjswan.envs.mdp.events import EventBinding
    from mjswan.envs.mdp.observations import ObservationBinding
    from mjswan.envs.mdp.terminations import TerminationBinding

    # Top-level mjswan namespace
    assert mjswan.ObsBinding is ObservationBinding
    assert mjswan.ObsFunc is ObservationBinding
    assert mjswan.TermBinding is TerminationBinding
    assert mjswan.TermFunc is TerminationBinding
    assert mjswan.EventFunc is EventBinding
    assert mjswan.MjlabMdpBinding is MdpBinding
    assert mjswan.CommandTermSpec is CommandBinding

    # Original leaf-module / package import paths
    from mjswan.command import CommandTermSpec
    from mjswan.envs.mdp import MjlabMdpBinding, ObsBinding, TermBinding
    from mjswan.envs.mdp.events import EventFunc
    from mjswan.envs.mdp.observations import ObsFunc
    from mjswan.envs.mdp.terminations import TermFunc

    assert ObsBinding is ObservationBinding
    assert ObsFunc is ObservationBinding
    assert TermBinding is TerminationBinding
    assert TermFunc is TerminationBinding
    assert EventFunc is EventBinding
    assert MjlabMdpBinding is MdpBinding
    assert CommandTermSpec is CommandBinding


@pytest.mark.parametrize(
    ("module_path", "old_fn"),
    [
        ("mjswan.envs.mdp.observations", "register_obs_func"),
        ("mjswan.envs.mdp.terminations", "register_termination_func"),
        ("mjswan.envs.mdp.events", "register_event_func"),
        ("mjswan.command", "register_command_term"),
    ],
)
def test_deprecated_register_fns_warn(module_path, old_fn):
    import importlib

    fn = getattr(importlib.import_module(module_path), old_fn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Register a harmless unsupported binding under a throwaway name.
        from mjswan.envs.mdp.observations import ObservationBinding

        fn("__compat_probe__", ObservationBinding(ts_name=""))
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
