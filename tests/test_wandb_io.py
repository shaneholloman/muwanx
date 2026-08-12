import pytest

from mjswan.wandb_io import resolve_wandb_artifact_path


class TestResolveWandbArtifactPath:
    def test_resolves_artifact_url_with_file_path(self):
        artifact_name, artifact_type, file_path = resolve_wandb_artifact_path(
            "https://wandb.ai/ttktjmt-org/csv_to_npz/artifacts/motions/"
            "mimickit_spinkick_safe/v0/files/motion.npz"
        )

        assert artifact_name == "ttktjmt-org/csv_to_npz/mimickit_spinkick_safe:v0"
        assert artifact_type == "motions"
        assert file_path == "motion.npz"

    def test_resolves_fully_qualified_artifact_name(self):
        artifact_name, artifact_type, file_path = resolve_wandb_artifact_path(
            "ttktjmt-org/csv_to_npz/mimickit_spinkick_safe:v0"
        )

        assert artifact_name == "ttktjmt-org/csv_to_npz/mimickit_spinkick_safe:v0"
        assert artifact_type == "motions"
        assert file_path == "motion.npz"


class TestQuietMjlabManagerTables:
    """The `.pt`-conversion env's MDP tables are the build env's, printed early.

    Two `ManagerBasedRlEnv`s exist per scene — one to convert checkpoints, one to trace
    term bodies — so mjlab's per-manager tables appeared twice and the second set read
    as a second, different environment.
    """

    @staticmethod
    def _mod():
        pytest.importorskip("mjlab")
        from mjlab.envs import manager_based_rl_env as mod

        return mod

    def test_tables_are_swallowed_on_success(self, capsys):
        from mjswan.wandb_io import _quiet_mjlab_manager_tables

        mod = self._mod()
        with _quiet_mjlab_manager_tables():
            mod.print_info("[INFO] <EventManager> contains 2 active terms.")
        assert capsys.readouterr().out == ""

    def test_they_are_re_emitted_when_construction_fails(self, capsys):
        """A failure mid-construction must not lose what mjlab said on the way."""
        from mjswan.wandb_io import _quiet_mjlab_manager_tables

        mod = self._mod()
        with pytest.raises(ValueError, match="nconmax"):
            with _quiet_mjlab_manager_tables():
                mod.print_info("[INFO] <ActionManager> contains 1 active terms.")
                raise ValueError("nconmax too small")
        assert "<ActionManager>" in capsys.readouterr().out

    def test_print_info_is_restored_afterwards(self):
        from mjswan.wandb_io import _quiet_mjlab_manager_tables

        mod = self._mod()
        before = mod.print_info
        with _quiet_mjlab_manager_tables():
            pass
        assert mod.print_info is before
