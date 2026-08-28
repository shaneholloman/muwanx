"""Where a traced graph lands, and what happens when two of them want one path.

The scoping is the fix; the guard is what says so out loud if it ever stops holding.
"""

from __future__ import annotations

import pytest

from mjswan._graph_io import onnx_ref, write_onnx


class TestOnnxRef:
    def test_unscoped_ref_is_kind_and_name(self):
        assert onnx_ref("event", "push_robot") == "event/push_robot.onnx"

    def test_a_scope_puts_the_graph_under_its_owner(self):
        assert onnx_ref("obs", "policy", "walk") == "walk/obs/policy.onnx"

    def test_two_policies_naming_one_group_get_two_paths(self):
        # The whole point: "policy" is the ONNX input name both networks read, so the
        # group name alone cannot separate them.
        assert onnx_ref("obs", "policy", "walk") != onnx_ref("obs", "policy", "crawl")


class TestWriteOnnx:
    def test_writes_the_bytes_at_the_ref(self, tmp_path):
        write_onnx(tmp_path, "walk/obs/policy.onnx", b"graph")
        assert (tmp_path / "walk" / "obs" / "policy.onnx").read_bytes() == b"graph"

    def test_rewriting_the_same_graph_passes_through(self, tmp_path):
        # One term traced twice within a build is not a collision.
        write_onnx(tmp_path, "obs/policy.onnx", b"graph")
        write_onnx(tmp_path, "obs/policy.onnx", b"graph")
        assert (tmp_path / "obs" / "policy.onnx").read_bytes() == b"graph"

    def test_a_different_graph_at_one_path_fails_the_build(self, tmp_path):
        write_onnx(tmp_path, "obs/policy.onnx", b"walk graph")
        with pytest.raises(ValueError, match="obs/policy.onnx"):
            write_onnx(tmp_path, "obs/policy.onnx", b"crawl graph")
        # The first writer's graph survives; the build stops rather than shipping either.
        assert (tmp_path / "obs" / "policy.onnx").read_bytes() == b"walk graph"
