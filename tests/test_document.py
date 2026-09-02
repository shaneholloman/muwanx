"""The `.swn` simulation document (ADR 0006 §8).

Layer: L1 — a built tree on disk, the Node build and template copy mocked. What is
pinned: a document is the manifest plus the project trees and nothing of the engine,
it unpacks to the tree it was made from, and publishing it uploads exactly what
publishing the directory would.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mjswan.app import MjswanApp
from mjswan.builder import Builder
from mjswan.document import (
    document_files,
    is_document,
    read_manifest,
    unpack_document,
    write_document,
)
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.publish import HttpResponse, plan_publish, publish_dist


@pytest.fixture
def built(tmp_path, minimal_model, minimal_onnx, monkeypatch) -> Path:
    """A real `_save_web` tree with the engine files a build would also carry."""
    monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
    monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())
    builder = Builder()
    scene = builder.add_project(name="Demo").add_scene(
        control_dt=0.02, name="Humanoid", model=minimal_model
    )
    scene.add_policy(
        name="walk",
        policy=minimal_onnx,
        actions={"joint_pos": JointPositionActionCfg(actuator_names=(".*",))},
    )
    out = tmp_path / "dist"
    builder._save_web(out)
    # What the SPA copy would have put there: the engine, not the document.
    (out / "index.html").write_text("<!doctype html>")
    (out / "assets" / "index-abc.js").write_text("console.log(1)")
    (out / "assets" / "mujoco.wasm").write_bytes(b"\\0asm")
    return out


class TestDocumentFiles:
    def test_manifest_first_then_every_file_under_a_project(self, built):
        files = [p.as_posix() for p in document_files(built)]
        assert files[0] == "manifest.json"
        assert "demo/humanoid/scene.mjb" in files
        assert "demo/humanoid/policy/walk.onnx" in files
        assert not any(f.startswith("assets/") or f == "index.html" for f in files)

    def test_a_tree_without_a_manifest_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            document_files(tmp_path)


class TestWriteAndUnpack:
    def test_writes_beside_the_directory_with_the_swn_suffix(self, built):
        path = write_document(built)
        assert path == built.with_suffix(".swn")
        assert is_document(path)
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        assert names[0] == "manifest.json"
        assert "demo/humanoid/policy/walk.onnx" in names
        assert "index.html" not in names

    def test_a_target_without_the_suffix_gets_it(self, built, tmp_path):
        assert write_document(built, tmp_path / "out" / "sim").suffix == ".swn"

    def test_unpacks_to_the_tree_it_was_made_from(self, built, tmp_path):
        path = write_document(built)
        tree = unpack_document(path, tmp_path / "unpacked")
        original = {
            p.as_posix(): (built / p).read_bytes() for p in document_files(built)
        }
        restored = {
            p.relative_to(tree).as_posix(): p.read_bytes()
            for p in tree.rglob("*")
            if p.is_file()
        }
        assert restored == original
        assert read_manifest(path) == read_manifest(tree)

    def test_already_compressed_assets_are_stored_not_deflated(self, built):
        (built / "demo" / "humanoid" / "assets").mkdir(exist_ok=True)
        (built / "demo" / "humanoid" / "assets" / "clip.npz").write_bytes(b"NPZ" * 8)
        path = write_document(built)
        with zipfile.ZipFile(path) as zf:
            by_name = {i.filename: i.compress_type for i in zf.infolist()}
        assert by_name["demo/humanoid/assets/clip.npz"] == zipfile.ZIP_STORED
        assert by_name["manifest.json"] == zipfile.ZIP_DEFLATED

    def test_an_archive_that_escapes_its_directory_is_refused(self, tmp_path):
        evil = tmp_path / "evil.swn"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("manifest.json", "{}")
            zf.writestr("../outside.txt", "x")
        with pytest.raises(ValueError, match="outside"):
            unpack_document(evil, tmp_path / "target")
        assert not (tmp_path / "outside.txt").exists()

    def test_a_zip_without_a_manifest_is_not_a_document(self, tmp_path):
        other = tmp_path / "other.swn"
        with zipfile.ZipFile(other, "w") as zf:
            zf.writestr("readme.txt", "hi")
        with pytest.raises(ValueError, match="not a simulation document"):
            unpack_document(other, tmp_path / "target")


class TestAppSaveDocument:
    def test_save_document_defaults_beside_the_build(self, built):
        path = MjswanApp(built).save_document()
        assert path == built.with_suffix(".swn") and path.is_file()


class _Transport:
    """Hands out one presigned URL per requested file and records every PUT."""

    def __init__(self):
        self.puts: list[str] = []

    def post_json(self, url, body, token):
        if url.endswith("/upload-session"):
            uploads = [
                {"path": f["path"], "url": f"https://r2.example.com/{f['path']}"}
                for f in body["manifest"]
            ]
            return HttpResponse(
                200, json.dumps({"upload_id": "u1", "uploads": uploads}).encode()
            )
        return HttpResponse(200, json.dumps({"id": "sim_1"}).encode())

    def put_bytes(self, url, data, content_type):
        self.puts.append(url.removeprefix("https://r2.example.com/"))
        return HttpResponse(200, b"")


class TestPublishingADocument:
    def test_uploads_the_same_file_set_as_the_directory(self, built, monkeypatch):
        monkeypatch.setenv("MJSWAN_TOKEN", "t")
        as_directory = {f.upload_path for f in plan_publish(built).files}
        path = write_document(built)
        transport = _Transport()
        result = publish_dist(path, transport=transport, api_base="https://api.test")
        assert result.id == "sim_1"
        assert set(transport.puts) == as_directory
        assert "manifest.json" in as_directory

    def test_a_bad_document_is_a_publish_error(self, tmp_path, monkeypatch):
        from mjswan.publish import PublishError

        monkeypatch.setenv("MJSWAN_TOKEN", "t")
        not_a_zip = tmp_path / "broken.swn"
        not_a_zip.write_bytes(b"definitely not a zip")
        with pytest.raises(PublishError, match="not a ZIP archive"):
            publish_dist(not_a_zip, transport=_Transport(), api_base="https://api.test")


class TestInfoCli:
    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_info_reads_a_document_like_a_directory(self, built):
        from mjswan._cli import app

        path = write_document(built)
        as_dir = self._runner().invoke(app, ["info", str(built)])
        as_doc = self._runner().invoke(app, ["info", str(path)])
        assert as_dir.exit_code == 0, as_dir.output
        assert as_doc.exit_code == 0, as_doc.output
        for output in (as_dir.output, as_doc.output):
            assert "Demo" in output and "Humanoid" in output and "walk" in output
        assert "mjswan app" in as_dir.output
        assert "mjswan document" in as_doc.output

    def test_info_refuses_a_document_without_a_manifest(self, tmp_path):
        from mjswan._cli import app

        other = tmp_path / "other.swn"
        with zipfile.ZipFile(other, "w") as zf:
            zf.writestr("readme.txt", "hi")
        result = self._runner().invoke(app, ["info", str(other)])
        assert result.exit_code == 1
        assert "not a simulation document" in result.output
