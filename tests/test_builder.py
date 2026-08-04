"""Tests for mjswan.Builder — project ID assignment, config JSON structure, and build output.

Layer breakdown:
  L1 (pure logic / lightweight I/O): TestProjectIdAssignment, TestBuilderValidation,
                                     TestSaveConfigJson, TestSaveWebPolicyJson
  L3 slow (triggers frontend build): TestFullBuild

Run only L1 tests (pre-commit):  pytest -m "not slow"
Run all tests (CI):               pytest
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import mjswan
from mjswan._build_client import ClientBuilder
from mjswan.builder import Builder
from mjswan.envs.mdp import events as evt_fns
from mjswan.envs.mdp import observations as obs_fns
from mjswan.envs.mdp import terminations as term_fns
from mjswan.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.utils import name2id


# ===========================================================================
# L1 — project ID assignment rules
# ===========================================================================
class TestProjectIdAssignment:
    def test_first_project_without_explicit_id_gets_none(self):
        builder = Builder()
        project = builder.add_project(name="Main Demo")
        assert project.id is None

    def test_second_project_without_explicit_id_gets_auto_id(self):
        builder = Builder()
        builder.add_project(name="Main Demo")
        second = builder.add_project(name="MuJoCo Menagerie")
        assert second.id == name2id("MuJoCo Menagerie")

    def test_auto_id_uses_name2id_transform(self):
        builder = Builder()
        builder.add_project(name="First")
        second = builder.add_project(name="My Project Name")
        assert second.id == "my_project_name"

    def test_explicit_id_used_as_is_on_first_project(self):
        project = Builder().add_project(name="Main Demo", id="custom")
        assert project.id == "custom"

    def test_explicit_id_used_as_is_on_subsequent_project(self):
        builder = Builder()
        builder.add_project(name="First")
        second = builder.add_project(name="Second", id="explicit_id")
        assert second.id == "explicit_id"

    def test_mixed_id_sequence(self):
        builder = Builder()
        p1 = builder.add_project(name="Project A")
        p2 = builder.add_project(name="Project B")
        p3 = builder.add_project(name="Project C", id="custom")
        assert p1.id is None
        assert p2.id == name2id("Project B")
        assert p3.id == "custom"

    def test_get_projects_returns_independent_copy(self):
        builder = Builder()
        builder.add_project(name="Test")
        copy = builder.get_projects()
        copy.clear()
        assert len(builder.get_projects()) == 1


# ===========================================================================
# L1 — GTM ID handling
# ===========================================================================
class TestBuilderGtmId:
    def test_defaults_to_none(self):
        assert Builder()._gtm_id is None

    def test_stored_when_provided(self):
        assert Builder(gtm_id="GTM-W79HQ38W")._gtm_id == "GTM-W79HQ38W"


class TestClientBuilderCustomTerms:
    """Custom terms are runtime plugins now (ADR 0004 §10): the engine gets empty
    Custom* stubs, and author TS is collected for esbuild into a standalone ESM."""

    def test_generate_empty_custom_stubs_writes_empty_registries(self, tmp_path):
        project_dir = tmp_path / "template"
        for sub in ("observation", "command", "termination", "event"):
            (project_dir / "src" / "core" / sub).mkdir(parents=True)

        ClientBuilder(project_dir).generate_empty_custom_stubs()

        core = project_dir / "src" / "core"
        obs = (core / "observation" / "custom_observations.ts").read_text()
        evt = (core / "event" / "custom_events.ts").read_text()
        assert "CustomObservations" in obs and "= {};" in obs
        assert "CustomCommands" in (core / "command" / "custom_commands.ts").read_text()
        assert (
            "CustomTerminations"
            in (core / "termination" / "custom_terminations.ts").read_text()
        )
        assert "CustomEvents" in evt and "= {};" in evt

    def test_collect_custom_terms_gathers_ts_src_by_kind(self, tmp_path, monkeypatch):
        src = tmp_path / "FooTerm.ts"
        src.write_text("export class FooTerm {}\n")
        monkeypatch.setattr(
            term_fns,
            "_custom_registry",
            {"foo": SimpleNamespace(ts_name="FooTerm", ts_src=str(src))},
        )
        terms = ClientBuilder._collect_custom_terms()
        assert terms["terminations"] == {"FooTerm": src.resolve()}

    def test_collect_custom_terms_skips_declarative_callable(self, monkeypatch):
        # A term traced to ONNX (ADR 0005) is a plain callable with no
        # ts_src — must be skipped, not crash.
        def base_lin_vel(env, **params):
            del env, params

        monkeypatch.setattr(obs_fns, "_custom_registry", {"base_lin_vel": base_lin_vel})
        assert "observations" not in ClientBuilder._collect_custom_terms()


class TestFrontendBuildCache:
    """The SPA is project-independent, so a matching dist/ is reused (no Node)."""

    def test_cache_matches_only_on_identical_meta(self, tmp_path):
        cb = ClientBuilder(tmp_path)
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html>")
        meta = cb._build_meta(base_path="/", gtm_id=None, mt=False, debug=False)
        (dist / ".mjswan-build-meta.json").write_text(json.dumps(meta))

        assert cb._cached_spa_matches(meta) is True
        # A different base_path must invalidate the cache.
        other = cb._build_meta(base_path="/sub/", gtm_id=None, mt=False, debug=False)
        assert cb._cached_spa_matches(other) is False

    def test_build_frontend_false_without_cache_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no matching prebuilt"):
            ClientBuilder(tmp_path).build(build_frontend=False)

    def test_source_change_invalidates_fingerprint(self, tmp_path):
        (tmp_path / "src").mkdir()
        app = tmp_path / "src" / "App.tsx"
        app.write_text("export default 1;\n")
        cb = ClientBuilder(tmp_path)
        before = cb._source_fingerprint()
        app.write_text("export default 2;\n")
        assert cb._source_fingerprint() != before

    def test_version_bump_alone_does_not_churn_fingerprint(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps({"name": "mjswan", "version": "0.7.0", "scripts": {}})
        )
        cb = ClientBuilder(tmp_path)
        before = cb._source_fingerprint()
        # Version is keyed separately and normalized out of the fingerprint.
        pkg.write_text(
            json.dumps({"name": "mjswan", "version": "0.8.0", "scripts": {}})
        )
        assert cb._source_fingerprint() == before


@pytest.mark.slow
class TestBuildPluginsModule:
    """esbuild-bundles author terms into a standalone ESM (needs Node/esbuild)."""

    def test_bundles_author_term_into_standalone_esm(self, tmp_path, monkeypatch):
        template = Path(mjswan.__file__).parent / "template"
        esbuild = template / "node_modules" / ".bin" / "esbuild"
        if not esbuild.exists():
            pytest.skip("esbuild not installed (run npm install in template)")

        src = tmp_path / "MyEvent.ts"
        src.write_text(
            "import { EventBase, type EventContext } from 'mjswan/event';\n"
            "export class MyEvent extends EventBase {\n"
            "  onReset(_ctx: EventContext): void {}\n"
            "}\n"
        )
        monkeypatch.setattr(
            evt_fns,
            "_custom_registry",
            {"my_event": SimpleNamespace(ts_name="MyEvent", ts_src=str(src))},
        )
        out = tmp_path / "plugins.js"
        assert ClientBuilder(template).build_plugins_module(out) is True
        code = out.read_text()
        assert (
            "MyEvent" in code and "events" in code and "export {" in code
        )  # grouped export
        assert "EventBase" in code  # base class bundled (self-contained)
        assert "mjswan/event" not in code  # no bare imports remain

    def test_term_importing_three_reuses_engine_instance(self, tmp_path, monkeypatch):
        # A term that imports `three` must resolve to the engine's single instance
        # (global), not a bundled duplicate — else instanceof / raycasting break.
        template = Path(mjswan.__file__).parent / "template"
        if not (template / "node_modules" / ".bin" / "esbuild").exists():
            pytest.skip("esbuild not installed (run npm install in template)")

        src = tmp_path / "ThreeObs.ts"
        src.write_text(
            "import * as THREE from 'three';\n"
            "import { EventBase, type EventContext } from 'mjswan/event';\n"
            "export class ThreeObs extends EventBase {\n"
            "  onReset(_ctx: EventContext): void { void new THREE.Vector3(); }\n"
            "}\n"
        )
        monkeypatch.setattr(
            evt_fns,
            "_custom_registry",
            {"three_obs": SimpleNamespace(ts_name="ThreeObs", ts_src=str(src))},
        )
        out = tmp_path / "plugins.js"
        assert ClientBuilder(template).build_plugins_module(out) is True
        code = out.read_text()
        assert "__mjswanThree" in code  # `three` resolved to the shared-instance shim
        assert "three" not in code.split("\n")[0]  # no bare `three` import survived


# ===========================================================================
# L1 — validation
# ===========================================================================
class TestBuilderValidation:
    def test_build_with_no_projects_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot build an empty application"):
            Builder().build(tmp_path / "out")

    def test_scene_with_a_policy_needs_a_control_dt(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        # The runtime derived this from a hardcoded 0.02 s, right for the locomotion and
        # manipulation tasks (0.005 x 4) and wrong for Cartpole (0.01 x 5), which played
        # 2.5x too fast in silence. Nothing about a wrong control rate raises at
        # playback, so the build refuses to guess.
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(name="Policy", policy=minimal_onnx)
        with pytest.raises(ValueError, match="has policies but no control_dt"):
            builder.build(tmp_path / "out", build_frontend=False)

    def test_scene_without_a_policy_needs_no_control_dt(self, tmp_path, minimal_model):
        # A viewer-only scene has no trained rate to match, so requiring one would be
        # noise. It must not raise for the reason above.
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        builder.build(tmp_path / "out", build_frontend=False)

    def test_non_positive_control_dt_is_rejected(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            name="S", model=minimal_model, control_dt=0.0
        )
        scene.add_policy(name="Policy", policy=minimal_onnx)
        with pytest.raises(ValueError, match="must be a positive number of seconds"):
            builder.build(tmp_path / "out", build_frontend=False)

    def test_control_dt_reaches_the_scene_entry(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            name="S", model=minimal_model, control_dt=0.05
        )
        scene.add_policy(name="Policy", policy=minimal_onnx)
        builder._save_config_json(tmp_path)
        config = json.loads((tmp_path / "assets" / "config.json").read_text())
        assert config["projects"][0]["scenes"][0]["controlDt"] == 0.05

    def test_policy_filename_rejects_empty_string(self):
        with pytest.raises(ValueError):
            Builder()._policy_filename("")

    def test_policy_filename_rejects_forward_slash(self):
        with pytest.raises(ValueError):
            Builder()._policy_filename("path/policy")

    def test_policy_filename_rejects_backslash(self):
        with pytest.raises(ValueError):
            Builder()._policy_filename("path\\policy")

    def test_policy_filename_accepts_plain_name(self):
        assert Builder()._policy_filename("my_policy") == "my_policy"


# ===========================================================================
# L1 — from_mjlab wiring (no mjlab/wandb required, dependencies monkeypatched)
# ===========================================================================
class TestFromMjlab:
    """Verify Builder.from_mjlab forwards run_path to SceneHandle.add_policy_wandb."""

    @staticmethod
    def _patch(monkeypatch):
        """Patch add_scene_mjlab to skip the real mjlab loader and return a mock SceneHandle."""
        from mjswan.project import ProjectHandle

        scene_handle = MagicMock(name="SceneHandle")
        monkeypatch.setattr(
            ProjectHandle,
            "add_scene_mjlab",
            lambda self, task_id, *, play=False: scene_handle,
        )
        return scene_handle

    def test_no_run_path_does_not_call_add_policy_wandb(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        Builder.from_mjlab("go2_flat")
        scene_handle.add_policy_wandb.assert_not_called()

    def test_str_run_path_forwarded_with_task_id(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        Builder.from_mjlab("go2_flat", run_path="org/proj/abc")
        scene_handle.add_policy_wandb.assert_called_once_with(
            "org/proj/abc", task_id="go2_flat"
        )

    def test_list_run_path_forwarded_as_is(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        run_paths = ["org/proj/a", "org/proj/b"]
        Builder.from_mjlab("go2_flat", run_path=run_paths)
        scene_handle.add_policy_wandb.assert_called_once_with(
            run_paths, task_id="go2_flat"
        )


# ===========================================================================
# L1 — _save_config_json output structure (no frontend build)
# ===========================================================================
class TestSaveConfigJson:
    def _read_config(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "assets" / "config.json").read_text())

    def test_config_contains_version(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        assert self._read_config(tmp_path)["version"] == mjswan.__version__

    def test_config_has_projects_list(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        config = self._read_config(tmp_path)
        assert isinstance(config["projects"], list)
        assert len(config["projects"]) == 1

    def test_project_name_and_id_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Main Demo").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        project = self._read_config(tmp_path)["projects"][0]
        assert project["name"] == "Main Demo"
        assert project["id"] is None

    def test_config_omits_plugins_when_declarative(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        config = self._read_config(tmp_path)
        assert config["uses_custom_js"] is False
        assert "plugins" not in config

    def test_config_references_plugins_when_custom_js(
        self, tmp_path, minimal_model, monkeypatch
    ):
        # A registered ts_src term flips uses_custom_js and adds the plugin ref.
        monkeypatch.setattr(
            evt_fns,
            "_custom_registry",
            {"e": SimpleNamespace(ts_src="x.ts", ts_name="E")},
        )
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        config = self._read_config(tmp_path)
        assert config["uses_custom_js"] is True
        assert config["plugins"] == "assets/plugins.js"

    def test_scene_path_uses_name2id_with_mjb_for_model(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="My Scene", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        scene = self._read_config(tmp_path)["projects"][0]["scenes"][0]
        assert scene["name"] == "My Scene"
        assert scene["path"] == "my_scene/scene.mjb"

    def test_scene_path_uses_mjz_for_spec(self, tmp_path, minimal_spec):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="My Scene", spec=minimal_spec
        )
        builder._save_config_json(tmp_path)
        scene = self._read_config(tmp_path)["projects"][0]["scenes"][0]
        assert scene["path"] == "my_scene/scene.mjz"

    def test_policy_without_config_path_has_no_config_key(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(name="Policy", policy=minimal_onnx)
        builder._save_config_json(tmp_path)
        policy = self._read_config(tmp_path)["projects"][0]["scenes"][0]["policies"][0]
        assert policy["name"] == "Policy"
        assert "config" not in policy

    def test_policy_motion_summary_is_included_in_root_config(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(name="Policy", policy=minimal_onnx).add_motion(
            name="Spin Kick",
            source="motion.npz",
            anchor_body_name="torso_link",
            body_names=("pelvis", "torso_link"),
            default=True,
        )

        builder._save_config_json(tmp_path)
        policy = self._read_config(tmp_path)["projects"][0]["scenes"][0]["policies"][0]
        assert policy["motions"] == [{"name": "Spin Kick", "default": True}]

    def test_multiple_projects_all_present_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Project A").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.add_project(name="Project B").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        projects = self._read_config(tmp_path)["projects"]
        assert len(projects) == 2
        assert projects[0]["name"] == "Project A"
        assert projects[1]["name"] == "Project B"

    def test_second_project_auto_id_reflected_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Main").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.add_project(name="MuJoCo Menagerie").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        projects = self._read_config(tmp_path)["projects"]
        assert projects[0]["id"] is None
        assert projects[1]["id"] == name2id("MuJoCo Menagerie")


# ===========================================================================
# L1 — uses_custom_js manifest flag (ADR 0003)
# ===========================================================================
class TestUsesCustomJsFlag:
    """Builds with any `ts_src`-bearing sentinel must mark themselves
    custom-JS so mjswan Cloud can refuse them.  Declarative-only builds must
    be marked clean.
    """

    def _read_config(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "assets" / "config.json").read_text())

    def _isolate_registries(self, monkeypatch):
        """Swap each MDP custom-registry for an empty dict so the test does
        not see registrations leaked in from other tests / modules."""
        from mjswan import command as command_mod
        from mjswan.envs.mdp import events as events_mod
        from mjswan.envs.mdp import observations as obs_mod
        from mjswan.envs.mdp import terminations as term_mod

        monkeypatch.setattr(obs_mod, "_custom_registry", {})
        monkeypatch.setattr(term_mod, "_custom_registry", {})
        monkeypatch.setattr(events_mod, "_custom_registry", {})
        monkeypatch.setattr(command_mod, "_custom_registry", {})

    def test_clean_build_is_false(self, tmp_path, minimal_model, monkeypatch):
        self._isolate_registries(monkeypatch)
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        assert self._read_config(tmp_path)["uses_custom_js"] is False

    def test_custom_obs_ts_src_flips_flag_true(
        self, tmp_path, minimal_model, monkeypatch
    ):
        self._isolate_registries(monkeypatch)
        from mjswan.envs.mdp import observations as obs_mod

        obs_mod._custom_registry["my_term"] = obs_mod.ObservationBinding(
            ts_name="MyTerm", ts_src="/tmp/whatever.ts"
        )
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        assert self._read_config(tmp_path)["uses_custom_js"] is True

    def test_custom_term_ts_src_flips_flag_true(
        self, tmp_path, minimal_model, monkeypatch
    ):
        self._isolate_registries(monkeypatch)
        from mjswan.envs.mdp import terminations as term_mod

        term_mod._custom_registry["my_term"] = term_mod.TerminationBinding(
            ts_name="MyTerm", ts_src="/tmp/whatever.ts"
        )
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        assert self._read_config(tmp_path)["uses_custom_js"] is True

    def test_declarative_override_does_not_flip_flag(
        self, tmp_path, minimal_model, monkeypatch
    ):
        """Registered sentinel without ts_src is a declarative param override —
        the build is still declarative-only."""
        self._isolate_registries(monkeypatch)
        from mjswan.envs.mdp import terminations as term_mod

        term_mod._custom_registry["out_of_terrain_bounds"] = (
            term_mod.TerminationBinding(
                ts_name="OutOfTerrainBounds",
                defaults={"limit_x": 5.0, "limit_y": 5.0},
            )
        )
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        assert self._read_config(tmp_path)["uses_custom_js"] is False


# NOTE: the transitional name-collision check (TestNoBuiltinNameShadowing) was
# removed along with the check itself — after ADR 0003 no MDP category keeps a
# named built-in, so a ts_src term cannot shadow one.


# ===========================================================================
# A live env for ONNX tracing (ADR 0005) — these helpers stand in for a
# task's own env when the scene is a raw `add_scene(model=...)` (no mjlab
# task, hence no `mjlab_env` normally). They satisfy the tracer's only
# contract, `env.scene[name].data.<field>`, with plain torch tensors; the
# functions below are "self-authored" observation/termination bodies (ADR
# 0005 point 2 — traced exactly like an mjlab function, no reimplementation).
# Requires torch; callers must `pytest.importorskip("torch")` first.
def _fake_trace_env():
    import torch

    class _Data:
        def __init__(self, **fields):
            for k, v in fields.items():
                setattr(self, k, v)

    class _Entity:
        def __init__(self, data):
            self.data = data

    class _Scene:
        def __init__(self, entities):
            self._entities = entities

        def __getitem__(self, name):
            return self._entities[name]

    class _ActionTerm:
        def __init__(self, raw_action):
            self.raw_action = raw_action

    class _ActionManager:
        """Two terms, so `last_action(action_name=…)` is a real slice.

        Every buildable mjlab task declares exactly one action term, where a term's
        slice and the whole vector coincide — so a single-term fake could not tell a
        correct `action_offset` from a missing one. `arm` takes [0,3) and `gripper`
        the tail, mirroring `ActionManager.process_action`'s split.
        """

        def __init__(self):
            self.action = torch.tensor([[1.0, 2.0, 3.0, 9.0]])
            self.active_terms = ["arm", "gripper"]
            self.action_term_dim = [3, 1]

        def get_term(self, name):
            offset = 0
            for term_name, dim in zip(self.active_terms, self.action_term_dim):
                if term_name == name:
                    return _ActionTerm(self.action[:, offset : offset + dim])
                offset += dim
            raise KeyError(name)

    class _Env:
        def __init__(self, entities):
            self.scene = _Scene(entities)
            self.action_manager = _ActionManager()

    data = _Data(
        joint_pos=torch.tensor([[0.1, 0.2]]),
        default_joint_pos=torch.tensor([[0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.5]]),
    )
    return _Env({"robot": _Entity(data)})


# Named `last_action`, not `_fake_last_action`: the tracer classifies a native
# observation by `func.__name__` (`NATIVE_OBSERVATION_FUNCS`), so the name is the thing
# under test here. Body mirrors mjlab's `envs/mdp/observations.py`.
def last_action(env, *, action_name=None, **_):
    if action_name is None:
        return env.action_manager.action
    return env.action_manager.get_term(action_name).raw_action


def _fake_joint_pos_rel(env, *, entity_name="robot", **_):
    d = env.scene[entity_name].data
    return d.joint_pos - d.default_joint_pos


def _fake_bad_orientation(env, *, limit_angle, entity_name="robot", **_):
    import torch

    pg = env.scene[entity_name].data.projected_gravity_b
    return torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)).abs() > limit_angle


def _fake_root_height_below_minimum(env, *, minimum_height, entity_name="robot", **_):
    return env.scene[entity_name].data.root_link_pos_w[:, 2] < minimum_height


def _fake_time_out(env, *, max_episode_length=1e9, **_):
    import torch

    del env, max_episode_length
    return torch.tensor([False])


# ===========================================================================
# L1 — _save_web: actions/terminations serialization into policy JSON
# ===========================================================================
class TestSaveWebPolicyJson:
    """Tests for _save_web: verify actions/terminations are emitted into the
    generated policy JSON, covering both the no-config_path and config_path
    branches.  The frontend build and template copy are mocked out so these
    tests remain fast (L1).
    """

    @pytest.fixture(autouse=True)
    def _no_frontend(self, monkeypatch):
        """Skip the Node.js frontend build and the large template copytree."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())

    def _run(self, builder: Builder, tmp_path: Path) -> Path:
        """Call _save_web and return the output directory."""
        out = tmp_path / "out"
        builder._save_web(out)
        return out

    def _policy_json(
        self,
        out: Path,
        policy_name: str,
        scene_name: str = "S",
        project_dir: str = "main",
    ) -> dict:
        scene_id = name2id(scene_name)
        policy_id = name2id(policy_name)
        path = out / project_dir / "assets" / scene_id / f"{policy_id}.json"
        return json.loads(path.read_text())

    def test_spec_scene_config_path_matches_written_file(self, tmp_path, minimal_spec):
        """`_save_web` frees `scene.spec` right after writing `scene.mjz`, and writes
        `config.json` after that — the recorded path must still be the `.mjz` on disk,
        or the app 404s on the scene it just shipped."""
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", spec=minimal_spec)
        out = self._run(builder, tmp_path)

        scene_path = json.loads((out / "assets" / "config.json").read_text())[
            "projects"
        ][0]["scenes"][0]["path"]
        assert scene_path == "s/scene.mjz"
        assert (out / "main" / "assets" / scene_path).is_file()

    def test_terms_without_a_trace_env_name_the_missing_call(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """A plain `add_scene` scene has no env to trace against until it is given one.

        The tracer used to reach `None.scene` and raise `AttributeError` from four
        frames down, which says nothing about the one line the author is missing.
        """

        def joint_pos(env):
            return env.scene["robot"].data.joint_pos

        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={"joint_pos": ObservationTermCfg(func=joint_pos)}
                )
            },
        )
        with pytest.raises(ValueError, match="set_trace_env"):
            self._run(builder, tmp_path)

    # -----------------------------------------------------------------------
    # no-config_path branch
    # -----------------------------------------------------------------------

    def test_no_config_path_actions_emitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), scale=0.5, use_default_offset=True
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" in data
        assert "joint_pos" in data["actions"]
        assert data["actions"]["joint_pos"]["type"] == "joint_position"
        assert data["actions"]["joint_pos"]["scale"] == 0.5

    def test_no_config_path_terminations_emitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "time_out": TerminationTermCfg(func=_fake_time_out, time_out=True),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "terminations" in data
        assert "time_out" in data["terminations"]
        # A term reading no dynamic entity state is classified native (ADR 0005).
        entry = data["terminations"]["time_out"]
        assert entry["native"] == "elapsed_s >= episode_length_s"
        assert entry["time_out"] is True

    def test_no_config_path_both_blocks_emitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "effort": JointEffortActionCfg(actuator_names=(".*",), scale=2.0),
            },
            terminations={
                "fallen": TerminationTermCfg(
                    func=_fake_bad_orientation,
                    params={"limit_angle": 1.2},
                ),
            },
        )
        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        assert "actions" in data
        assert "terminations" in data
        assert data["actions"]["effort"]["type"] == "torque"
        # A term reading dynamic entity state is traced to ONNX (ADR 0005).
        fallen = data["terminations"]["fallen"]
        assert fallen["onnx"] == "term/fallen.onnx"
        assert (out / "main" / "assets" / "s" / fallen["onnx"]).exists()

    def test_no_config_path_actions_absent_when_not_set(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "time_out": TerminationTermCfg(func=_fake_time_out, time_out=True),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" not in data

    def test_no_config_path_terminations_absent_when_not_set(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(actuator_names=(".*",)),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "terminations" not in data

    def test_no_config_path_no_json_without_mdp_components(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(name="Policy", policy=minimal_onnx)
        out = self._run(builder, tmp_path)
        policy_id = name2id("Policy")
        scene_id = name2id("S")
        json_path = out / "main" / "assets" / scene_id / f"{policy_id}.json"
        assert not json_path.exists()

    def test_no_config_path_onnx_path_in_json(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"effort": JointEffortActionCfg(actuator_names=(".*",))},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["onnx"]["path"] == "policy.onnx"

    def test_no_config_path_motions_emitted_and_copied(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        motion_file = tmp_path / "spin_kick.npz"
        motion_file.write_bytes(b"motion-bytes")

        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"joint_pos": JointPositionActionCfg(actuator_names=(".*",))},
        ).add_motion(
            name="Spin Kick",
            source=str(motion_file),
            anchor_body_name="torso_link",
            body_names=("pelvis", "torso_link"),
            dataset_joint_names=["joint_a"],
            default=True,
        )

        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        assert data["motions"] == [
            {
                "name": "Spin Kick",
                "path": "policy_spin_kick.npz",
                "fps": 50.0,
                "anchor_body_name": "torso_link",
                "body_names": ["pelvis", "torso_link"],
                "dataset_joint_names": ["joint_a"],
                "default": True,
            }
        ]
        motion_out = out / "main" / "assets" / "s" / "policy_spin_kick.npz"
        assert motion_out.read_bytes() == b"motion-bytes"

    def test_no_config_path_commands_emitted_as_command_terms(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            commands={"velocity": mjswan.velocity_command()},
        )

        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["commands"]["velocity"]["name"] == "UiCommand"
        assert len(data["commands"]["velocity"]["ui"]["inputs"]) == 3
        assert data["commands"]["velocity"]["ui"]["inputs"][0]["name"] == "lin_vel_x"

    def test_plain_observation_term_traces_against_live_env(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        # A plain-callable observation func (mjlab's own, or a self-authored
        # one — same treatment per ADR 0005) is traced to ONNX against the
        # scene's live env; no mjswan-side reimplementation is involved.
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={
                        "joint_pos": ObservationTermCfg(
                            func=_fake_joint_pos_rel, scale=0.5
                        ),
                    }
                ),
            },
        )

        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        # The group fuses (ADR 0005 §4): one graph named for the group, not one per
        # term, and `scale` is folded into it rather than shipped for the runtime.
        group = data["observations"]["policy"]
        assert group["fused"] == "obs/policy.onnx"
        assert group["layout"] == [{"name": "joint_pos", "size": 2}]
        assert group["size"] == 2
        assert "scale" not in group
        assert (out / "main" / "assets" / "s" / group["fused"]).exists()

    def test_last_action_with_a_term_name_emits_its_slice_offset(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        # The two-action-term shape, through the real Builder. No mjlab task has it —
        # all four declare one term — so without this the offset was only ever checked
        # against a stub action manager, never emitted by a build.
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={
                        "joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel),
                        "gripper_action": ObservationTermCfg(
                            func=last_action, params={"action_name": "gripper"}
                        ),
                    }
                ),
            },
        )

        group = self._policy_json(self._run(builder, tmp_path), "Policy")[
            "observations"
        ]["policy"]
        native = next(
            entry
            for entry in group["native_inputs"]
            if entry["name"] == "gripper_action"
        )
        assert native["native"] == "prev_action"
        assert native["action_name"] == "gripper"
        # `arm` holds [0,3), so `gripper` starts at 3. Without the offset the runtime
        # feeds the graph `arm`'s first element at `gripper`'s width — the right width,
        # the wrong term.
        assert native["action_offset"] == 3
        assert native["size"] == 1
        # And the group still fuses around it: the native term is a graph *input*.
        assert group["layout"] == [
            {"name": "joint_pos", "size": 2},
            {"name": "gripper_action", "size": 1},
        ]

    def test_last_action_naming_no_action_term_fails_the_build(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        # Degrading would hand the runtime the whole action vector's head, which is the
        # silently-wrong observation the offset exists to prevent.
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={
                        "joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel),
                        "grip": ObservationTermCfg(
                            func=last_action, params={"action_name": "grippr"}
                        ),
                    }
                ),
            },
        )
        with pytest.raises(ValueError, match=r"does not define.*arm, gripper"):
            self._run(builder, tmp_path)

    def test_per_term_observations_when_the_group_cannot_fuse(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        # A term stacking its own history keeps the per-term path: mjlab stacks
        # before concatenating, so one fused output would order the group's history
        # differently (see `_group_is_fusable`).
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={
                        "joint_pos": ObservationTermCfg(
                            func=_fake_joint_pos_rel, scale=0.5, history_length=3
                        ),
                    }
                ),
            },
        )

        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        terms = data["observations"]["policy"]
        assert isinstance(terms, list)
        assert terms[0]["onnx"] == "obs/joint_pos.onnx"
        assert terms[0]["scale"] == 0.5
        assert terms[0]["history_length"] == 3

    # -----------------------------------------------------------------------
    # config_path branch
    # -----------------------------------------------------------------------

    def test_config_path_actions_merged_into_existing_config(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps({"onnx": {"path": "old.onnx"}, "existing_key": "kept"})
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), use_default_offset=True
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" in data
        assert data["actions"]["joint_pos"]["type"] == "joint_position"
        assert data["existing_key"] == "kept"

    def test_config_path_terminations_merged_into_existing_config(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps({"onnx": {"path": "old.onnx"}, "existing_key": "kept"})
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            terminations={
                "fallen": TerminationTermCfg(
                    func=_fake_bad_orientation,
                    params={"limit_angle": 0.8},
                ),
            },
        )
        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        assert "terminations" in data
        # A term reading dynamic entity state is traced to ONNX (ADR 0005).
        fallen = data["terminations"]["fallen"]
        assert fallen["onnx"] == "term/fallen.onnx"
        assert (out / "main" / "assets" / "s" / fallen["onnx"]).exists()
        assert data["existing_key"] == "kept"

    def test_config_path_both_blocks_merged(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "old.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "effort": JointEffortActionCfg(actuator_names=(".*",), scale=1.5),
            },
            terminations={
                "height": TerminationTermCfg(
                    func=_fake_root_height_below_minimum,
                    params={"minimum_height": 0.3},
                ),
            },
        )
        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        assert data["actions"]["effort"]["type"] == "torque"
        assert data["actions"]["effort"]["scale"] == 1.5
        # A term reading dynamic entity state is traced to ONNX (ADR 0005).
        height_entry = data["terminations"]["height"]
        assert height_entry["onnx"] == "term/height.onnx"
        assert (out / "main" / "assets" / "s" / height_entry["onnx"]).exists()

    def test_config_path_overwrites_existing_actions_block(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """actions from policy.actions fully replaces any pre-existing actions block."""
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps(
                {
                    "onnx": {"path": "old.onnx"},
                    "actions": {"old_action": {"type": "joint_position"}},
                }
            )
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "new_action": JointPositionActionCfg(actuator_names=(".*",)),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "new_action" in data["actions"]
        assert "old_action" not in data["actions"]

    def test_config_path_action_fields_survive_a_partial_override(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """A term names what it changes; the rest of the authored entry stays.

        For a motor-actuator robot the PD gains live in the policy's own config —
        the model has none — so a scene that only wants a different offset must not
        have to restate them.
        """
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps(
                {
                    "onnx": {"path": "old.onnx"},
                    "actions": {
                        "joint_pos": {
                            "type": "joint_position",
                            "scale": {"j": 0.5},
                            "stiffness": {"j": 40.0},
                            "damping": {"j": 2.5},
                        }
                    },
                }
            )
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), offset={"j": 0.25}
                )
            },
        )
        entry = self._policy_json(self._run(builder, tmp_path), "Policy")["actions"][
            "joint_pos"
        ]
        assert entry["offset"] == {"j": 0.25}
        assert entry["stiffness"] == {"j": 40.0}
        assert entry["damping"] == {"j": 2.5}
        assert entry["scale"] == {"j": 0.5}

    def test_config_path_onnx_path_updated(self, tmp_path, minimal_model, minimal_onnx):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "stale.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={"joint_pos": JointPositionActionCfg(actuator_names=(".*",))},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["onnx"]["path"] == "policy.onnx"

    def test_config_path_onnx_path_updated_when_extras_present(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "stale.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            extras={"model_overrides": {"geom_friction": [1.0, 0.5, 0.25]}},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["onnx"]["path"] == "policy.onnx"
        assert data["extras"] == {
            "model_overrides": {"geom_friction": [1.0, 0.5, 0.25]}
        }

    def test_config_path_actions_absent_from_json_when_not_set(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "old.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" not in data
        assert "terminations" not in data

    # -----------------------------------------------------------------------
    # Serialization correctness
    # -----------------------------------------------------------------------

    def test_joint_effort_action_scale_serialized(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"effort": JointEffortActionCfg(actuator_names=(".*",), scale=3.0)},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["actions"]["effort"] == {
            "type": "torque",
            "scale": 3.0,
            "actuator_names": [".*"],
        }

    def test_joint_position_default_offset_serialized(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), use_default_offset=False
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["actions"]["joint_pos"]["use_default_offset"] is False

    def test_timeout_termination_time_out_flag(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "time_out": TerminationTermCfg(func=_fake_time_out, time_out=True),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        term = data["terminations"]["time_out"]
        # A term reading no dynamic entity state is classified native (ADR 0005).
        assert term["native"] == "elapsed_s >= episode_length_s"
        assert term.get("time_out") is True

    def test_bad_orientation_params_serialized(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "fallen": TerminationTermCfg(
                    func=_fake_bad_orientation,
                    params={"limit_angle": 1.57},
                ),
            },
        )
        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        # A term reading dynamic entity state is traced to ONNX (ADR 0005);
        # `limit_angle` is closed over by the traced function, not serialized.
        fallen = data["terminations"]["fallen"]
        assert fallen["onnx"] == "term/fallen.onnx"
        assert (out / "main" / "assets" / "s" / fallen["onnx"]).exists()
        assert "time_out" not in fallen


# ===========================================================================
# L3 slow — full build pipeline (triggers frontend compilation)
# Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuild:
    def test_build_creates_assets_config_json(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "assets" / "config.json").exists()

    def test_build_with_model_creates_mjb_file(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        scene_dir = tmp_path / "out" / "main" / "assets" / "scene"
        assert (scene_dir / "scene.mjb").exists()

    def test_build_with_spec_creates_mjz_file(self, tmp_path, minimal_spec):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", spec=minimal_spec
        )
        builder.build(tmp_path / "out")
        scene_dir = tmp_path / "out" / "main" / "assets" / "scene"
        assert (scene_dir / "scene.mjz").exists()

    def test_build_project_without_id_uses_main_directory(
        self, tmp_path, minimal_model
    ):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "main").is_dir()

    def test_build_project_with_id_uses_id_as_directory(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test", id="demo").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "demo").is_dir()

    def test_build_returns_mjswan_app_instance(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        app = builder.build(tmp_path / "out")
        assert isinstance(app, mjswan.MjswanApp)

    def test_build_output_excludes_dev_and_test_files(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        out = tmp_path / "out"
        builder.build(out)
        # Dev/test scaffolding (harness, e2e, configs, type shims, fixtures) must
        # not ship in the published SPA.
        for leaked in (
            "e2e",
            "harness.html",
            "fixtures",
            "vite.config.ts",
            "vite.lib.config.ts",
            "vite.manifest.config.ts",
            "vitest.config.ts",
            "playwright.config.ts",
            "lib.d.ts",
            "manifest.d.ts",
            "src",
            "node_modules",
            "package.json",
        ):
            assert not (out / leaked).exists(), f"dev file leaked into build: {leaked}"
        assert (out / "index.html").exists() and (
            out / "assets" / "config.json"
        ).exists()


# ===========================================================================
# L1 — mt parameter: _save_mt_headers / no-headers when mt=False
# ===========================================================================
class TestMtHeaders:
    def test_mt_defaults_to_false(self):
        assert Builder()._mt is False

    def test_mt_true_stored(self):
        assert Builder(mt=True)._mt is True

    def test_save_mt_headers_creates_headers_file(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        assert (tmp_path / "_headers").exists()

    def test_save_mt_headers_contains_coop(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        content = (tmp_path / "_headers").read_text()
        assert "Cross-Origin-Opener-Policy: same-origin" in content

    def test_save_mt_headers_contains_coep(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        content = (tmp_path / "_headers").read_text()
        assert "Cross-Origin-Embedder-Policy: require-corp" in content

    def test_save_mt_headers_applies_wildcard_route(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        content = (tmp_path / "_headers").read_text()
        assert content.startswith("/*")

    def test_mt_false_does_not_write_headers(
        self, tmp_path, minimal_model, monkeypatch
    ):
        """_save_web with mt=False must not create _headers."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())
        builder = Builder(mt=False)
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        out = tmp_path / "out"
        builder._save_web(out)
        assert not (out / "_headers").exists()

    def test_mt_true_writes_headers(self, tmp_path, minimal_model, monkeypatch):
        """_save_web with mt=True must create _headers with COOP/COEP content."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())
        builder = Builder(mt=True)
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        out = tmp_path / "out"
        builder._save_web(out)
        headers_file = out / "_headers"
        assert headers_file.exists()
        content = headers_file.read_text()
        assert "Cross-Origin-Opener-Policy: same-origin" in content
        assert "Cross-Origin-Embedder-Policy: require-corp" in content

    def test_save_web_excludes_mt_template_dir(
        self, tmp_path, minimal_model, monkeypatch
    ):
        """_save_web must not copy the template-only _mt directory into output.

        The output is assembled allowlist-style from the built dist/ (+ LICENSE),
        so template-root scaffolding like _mt is excluded by construction.
        """
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())

        builder = Builder(mt=False)
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        out = tmp_path / "out"
        builder._save_web(out)

        assert not (out / "_mt").exists()


# ===========================================================================
# L3 slow — Phase 4: cache reuse + custom-JS runtime plugin module
# Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuildPhase4:
    def test_second_build_reuses_cached_frontend(self, tmp_path, minimal_model, capsys):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.build(tmp_path / "out1")  # warms the cache
        capsys.readouterr()
        builder2 = Builder()
        builder2.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder2.build(tmp_path / "out2")  # identical inputs → no frontend rebuild
        assert "Reusing cached frontend build" in capsys.readouterr().out
        assert (tmp_path / "out2" / "assets" / "config.json").exists()

    def test_custom_js_build_emits_plugins_module(
        self, tmp_path, minimal_model, monkeypatch
    ):
        from mjswan.envs.mdp import events as evt_mod

        template = Path(mjswan.__file__).parent / "template"
        if not (template / "node_modules" / ".bin" / "esbuild").exists():
            pytest.skip("esbuild not installed (run npm install in template)")

        term = tmp_path / "MyEvent.ts"
        term.write_text(
            "import { EventBase, type EventContext } from 'mjswan/event';\n"
            "export class MyEvent extends EventBase {\n"
            "  onReset(_ctx: EventContext): void {}\n"
            "}\n"
        )
        monkeypatch.setattr(
            evt_mod,
            "_custom_registry",
            {"my_event": SimpleNamespace(ts_name="MyEvent", ts_src=str(term))},
        )
        out = tmp_path / "out"
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.build(out)

        config = json.loads((out / "assets" / "config.json").read_text())
        assert config["uses_custom_js"] is True
        assert config["plugins"] == "assets/plugins.js"
        plugins = (out / "assets" / "plugins.js").read_text()
        assert "MyEvent" in plugins and "events" in plugins
        assert "mjswan/event" not in plugins  # standalone (base class bundled)


# ===========================================================================
# L3 slow — full mt=True build (triggers frontend compilation)
# Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuildMt:
    def test_mt_false_no_headers_file(self, tmp_path, minimal_model):
        builder = Builder(mt=False)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert not (tmp_path / "out" / "_headers").exists()

    def test_mt_false_no_coi_serviceworker(self, tmp_path, minimal_model):
        builder = Builder(mt=False)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert not (tmp_path / "out" / "coi-serviceworker.js").exists()

    def test_mt_true_writes_headers_file(self, tmp_path, minimal_model):
        builder = Builder(mt=True)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        headers = tmp_path / "out" / "_headers"
        assert headers.exists()
        content = headers.read_text()
        assert "Cross-Origin-Opener-Policy: same-origin" in content
        assert "Cross-Origin-Embedder-Policy: require-corp" in content

    def test_mt_true_emits_coi_serviceworker(self, tmp_path, minimal_model):
        builder = Builder(mt=True)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "coi-serviceworker.js").exists()

    def test_mt_true_injects_sw_script_into_html(self, tmp_path, minimal_model):
        builder = Builder(mt=True)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "coi-serviceworker.js" in html
        assert "crossOriginIsolated" in html


@pytest.mark.slow
class TestFullBuildGtmId:
    def test_gtm_snippet_injected_into_all_html_files(self, tmp_path, minimal_model):
        builder = Builder(gtm_id="GTM-SAMPLE123")
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        out = tmp_path / "out"
        for html_file in [out / "index.html", out / "main" / "index.html"]:
            html = html_file.read_text()
            assert "GTM-SAMPLE123" in html
            assert "googletagmanager.com/gtm.js" in html  # <head> script
            assert "googletagmanager.com/ns.html" in html  # <body> noscript

    def test_no_gtm_without_gtm_id(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "googletagmanager.com" not in html
