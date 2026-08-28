"""Where a traced MDP graph lands in the bundle, and the one write that puts it there.

The path *is* how a config refers to the graph: the browser resolves it relative to the
file that names it (``manifest/index.ts``'s ``siblingOf``), so the ref string and the
location on disk are one thing said once.

Kept free of the tracer, and so of torch, since both ends of a build reach for it.
"""

from __future__ import annotations

from pathlib import Path


def onnx_ref(kind: str, name: str, scope: str | None = None) -> str:
    """Bundle-relative path for a traced term's ``.onnx`` file.

    *scope* is the owning policy's id. A term or group name is unique within a policy
    but not within a scene, and every policy names its observation group ``"policy"`` —
    the ONNX input name its network reads (``mjlab_adapter.DEFAULT_OBS_GROUP_KEY``), not
    a label a policy is free to change — so unscoped, two policies in one scene write to
    the same file. Scene-scoped graphs (events) pass no scope.
    """
    ref = f"{kind}/{name}.onnx"
    return ref if scope is None else f"{scope}/{ref}"


def write_onnx(out_dir: Path, ref: str, onnx_bytes: bytes) -> None:
    """Write a traced graph, refusing to replace a different graph already at *ref*.

    A build wipes its output first, so an existing file is always this build's: identical
    bytes are one term traced twice and pass through, different bytes mean two owners
    resolved to one path. Left alone that is silent — the last write wins, and every
    config still naming the first loads the wrong graph without any width to catch it.
    """
    path = out_dir / ref
    if path.is_file() and path.read_bytes() != onnx_bytes:
        raise ValueError(
            f"Two different graphs both want {ref!r} under {out_dir}. One path cannot "
            "carry both: the second write wins and every config still pointing at the "
            "first would load the wrong graph, with no error at playback."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_bytes)


__all__ = ["onnx_ref", "write_onnx"]
