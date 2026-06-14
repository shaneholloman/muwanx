"""Publish a built mjswan ``dist/`` to mjswan Cloud (v2).

This module extracts the **data files** from a local build (config.json plus
scene/policy/motion/splat assets — never HTML/JS/WASM) and uploads them to
mjswan Cloud using the three-step presigned-upload protocol described in
mjswan-cloud ADR 0001 §6:

1. ``POST {base}/api/simulations/upload-session`` with a file manifest and the
   parsed ``config.json`` → presigned ``PUT`` URLs.
2. ``PUT`` each file's bytes to its presigned URL.
3. ``POST {base}/api/simulations/commit`` with the ``upload_id`` → the sim id.

The platform renders only **declarative** builds (no author-supplied code), so
``publish`` refuses any build whose ``config.json`` reports
``uses_custom_js: true`` — a fast, local UX gate that prevents a guaranteed
server-side rejection and the broken render it would otherwise produce. See
mjswan ADR 0003 for the declarative/custom-JS split and the ``uses_custom_js``
marker.

The upload root layout mirrors the build's ``dist/`` tree minus everything but
data files, with the manifest ``config.json`` hoisted to the upload root so the
engine's ``mount(element, configUrl)`` can resolve every other asset relative to
``configUrl``'s directory.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── Client-side constraints (mirror the server's; fail fast and locally) ──────

#: File extensions uploaded as simulation *data*. Everything else in ``dist/``
#: (``.html`` / ``.js`` / ``.css`` / ``.wasm`` / fonts / images) is the engine
#: shell and is never sent — the platform loads a pinned engine from its CDN.
DATA_EXTENSIONS: frozenset[str] = frozenset(
    {".json", ".mjz", ".onnx", ".npz", ".ply", ".splat"}
)

#: App-shell files that happen to share a data extension but are not simulation
#: data (the PWA manifest references the engine's own icons).
_EXCLUDED_BASENAMES: frozenset[str] = frozenset({"manifest.json"})

MAX_FILE_BYTES: int = 50 * 1024 * 1024
MAX_TOTAL_BYTES: int = 200 * 1024 * 1024
MAX_FILES: int = 64

DEFAULT_API_BASE: str = "https://api.mjswan.com"
TOKEN_ENV_VAR: str = "MJSWAN_TOKEN"

_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".mjz": "application/zip",
    ".onnx": "application/octet-stream",
    ".npz": "application/octet-stream",
    ".ply": "application/octet-stream",
    ".splat": "application/octet-stream",
}


def _content_type_for(path: str) -> str:
    return _CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


class PublishError(Exception):
    """Raised when a publish cannot proceed (validation or server rejection).

    Carries an optional ``file`` so a 422 server response of the shape
    ``{"error", "file"}`` can be surfaced verbatim to the user.
    """

    def __init__(self, message: str, *, file: str | None = None) -> None:
        super().__init__(message)
        self.file = file


# ── HTTP transport (injectable so tests need no network) ──────────────────────


@dataclass
class HttpResponse:
    status: int
    body: bytes

    def json(self) -> dict:
        if not self.body:
            return {}
        return json.loads(self.body.decode("utf-8"))


class HttpTransport:
    """Minimal stdlib HTTP transport. Replaceable in tests.

    Returns an :class:`HttpResponse` for any status code (including 4xx/5xx)
    rather than raising, so callers can read structured error bodies.
    """

    def post_json(self, url: str, body: dict, token: str) -> HttpResponse:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        return self._send(req)

    def put_bytes(self, url: str, data: bytes, content_type: str) -> HttpResponse:
        req = urllib.request.Request(
            url,
            data=data,
            method="PUT",
            headers={"Content-Type": content_type},
        )
        return self._send(req)

    @staticmethod
    def _send(req: urllib.request.Request) -> HttpResponse:
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted base URL)
                return HttpResponse(status=resp.status, body=resp.read())
        except urllib.error.HTTPError as exc:
            # 4xx/5xx — read the structured error body instead of raising.
            return HttpResponse(status=exc.code, body=exc.read())
        except urllib.error.URLError as exc:
            raise PublishError(f"Network error contacting mjswan Cloud: {exc.reason}")


# ── File collection ───────────────────────────────────────────────────────────


@dataclass
class _DistFile:
    """A data file selected for upload."""

    upload_path: str  # POSIX path within the upload root (no leading slash)
    source: Path  # absolute path on disk
    size: int


@dataclass
class PublishPlan:
    """Everything resolved locally before any network call is made."""

    dist_dir: Path
    config: dict
    files: list[_DistFile] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def manifest(self) -> list[dict]:
        return [
            {
                "path": f.upload_path,
                "contentType": _content_type_for(f.upload_path),
                "size": f.size,
            }
            for f in self.files
        ]


def _find_config(dist_dir: Path) -> Path:
    """Locate the build's manifest ``config.json``.

    The builder writes it to ``<dist>/assets/config.json``; fall back to a
    top-level ``config.json`` so a flattened upload tree also works.
    """
    candidates = [dist_dir / "assets" / "config.json", dist_dir / "config.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise PublishError(
        f"No config.json found in {dist_dir} "
        "(looked in assets/config.json and config.json). "
        "Pass the directory produced by builder.build()."
    )


def plan_publish(dist_dir: Path) -> PublishPlan:
    """Validate ``dist_dir`` and build the upload plan without any network I/O.

    Raises :class:`PublishError` on any client-side constraint violation:
    missing/invalid config, custom-JS build, too many/too-large files, or a
    path that escapes the upload root.
    """
    dist_dir = Path(dist_dir).expanduser().resolve()
    if not dist_dir.is_dir():
        raise PublishError(f"Not a directory: {dist_dir}")

    config_path = _find_config(dist_dir)
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise PublishError(f"Invalid config.json: {exc}")

    if config.get("uses_custom_js") is True:
        raise PublishError(
            "This build uses custom-JS MDP terms (uses_custom_js: true) and "
            "cannot be published to mjswan Cloud, which renders only declarative "
            "builds. Re-author the custom terms declaratively, or request the "
            "missing capability as an engine built-in. See mjswan ADR 0003.",
            file="config.json",
        )

    plan = PublishPlan(dist_dir=dist_dir, config=config)

    # The manifest config.json is hoisted to the upload root as "config.json"
    # so mount(configUrl) resolves every other asset relative to configUrl's
    # directory.
    plan.files.append(
        _DistFile(
            upload_path="config.json",
            source=config_path,
            size=config_path.stat().st_size,
        )
    )

    for path in sorted(dist_dir.rglob("*")):
        if not path.is_file():
            continue
        if path == config_path:
            continue  # already added at the upload root
        if path.suffix.lower() not in DATA_EXTENSIONS:
            continue
        if path.name in _EXCLUDED_BASENAMES:
            continue
        rel = path.relative_to(dist_dir).as_posix()
        _check_safe_path(rel)
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise PublishError(
                f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)}MB limit: "
                f"{rel} ({size} bytes)",
                file=rel,
            )
        plan.files.append(_DistFile(upload_path=rel, source=path, size=size))

    _validate_plan(plan)
    return plan


def _check_safe_path(rel: str) -> None:
    if rel.startswith("/") or rel.startswith("\\"):
        raise PublishError(f"Absolute paths are not allowed: {rel}", file=rel)
    parts = Path(rel).parts
    if ".." in parts:
        raise PublishError(f"Path traversal is not allowed: {rel}", file=rel)


def _validate_plan(plan: PublishPlan) -> None:
    if len(plan.files) > MAX_FILES:
        raise PublishError(
            f"Too many files: {len(plan.files)} (max {MAX_FILES}). "
            "Reduce the number of scenes/policies/motions in this build."
        )
    if plan.total_bytes > MAX_TOTAL_BYTES:
        raise PublishError(
            f"Total upload size {plan.total_bytes} bytes exceeds the "
            f"{MAX_TOTAL_BYTES // (1024 * 1024)}MB limit."
        )


# ── Token resolution ───────────────────────────────────────────────────────────


def resolve_token(token: str | None) -> str:
    """Resolve the Supabase access token (GitHub OAuth).

    Order: explicit ``token`` argument → ``MJSWAN_TOKEN`` environment variable.
    An interactive login flow is not implemented here; the error explains how
    to obtain a token.
    """
    resolved = token or os.environ.get(TOKEN_ENV_VAR)
    if not resolved:
        raise PublishError(
            "No mjswan Cloud access token found. Set the "
            f"{TOKEN_ENV_VAR} environment variable to a Supabase access token "
            "(GitHub OAuth), or pass token=... explicitly."
        )
    return resolved


# ── The publish flow ───────────────────────────────────────────────────────────


@dataclass
class PublishResult:
    id: str
    sim_id: str
    upload_id: str


def publish_dist(
    dist_dir: str | Path,
    *,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    token: str | None = None,
    api_base: str = DEFAULT_API_BASE,
    transport: HttpTransport | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> PublishResult:
    """Publish a built ``dist/`` directory to mjswan Cloud.

    Args:
        dist_dir: Path to a directory produced by ``builder.build()``.
        title: Simulation title. Defaults to the first project's name.
        description: Optional description.
        tags: Optional list of tags.
        token: Supabase access token. Falls back to ``$MJSWAN_TOKEN``.
        api_base: Cloud API base URL. Defaults to ``https://api.mjswan.com``.
        transport: HTTP transport (injectable for tests).
        on_progress: Optional callback invoked with human-readable status lines.

    Returns:
        :class:`PublishResult` with the published simulation id.

    Raises:
        PublishError: on any client-side validation failure or server rejection.
    """
    transport = transport or HttpTransport()
    base = api_base.rstrip("/")
    notify = on_progress or (lambda _msg: None)

    plan = plan_publish(Path(dist_dir))
    resolved_token = resolve_token(token)

    resolved_title = title or _default_title(plan.config)

    body: dict = {
        "title": resolved_title,
        "manifest": plan.manifest(),
        "config": plan.config,
    }
    if description is not None:
        body["description"] = description
    if tags:
        body["tags"] = tags

    notify(f"Requesting upload session for {len(plan.files)} file(s)…")
    session_resp = transport.post_json(
        f"{base}/api/simulations/upload-session", body, resolved_token
    )
    _raise_for_status(session_resp, "upload-session")
    session = session_resp.json()

    upload_id = session.get("upload_id")
    if not upload_id:
        raise PublishError("upload-session response missing upload_id.")
    uploads = {u["path"]: u["url"] for u in session.get("uploads", [])}

    by_path = {f.upload_path: f for f in plan.files}
    for path, url in uploads.items():
        f = by_path.get(path)
        if f is None:
            raise PublishError(
                f"Server requested upload for unknown file: {path}", file=path
            )
        notify(f"Uploading {path} ({f.size} bytes)…")
        put_resp = transport.put_bytes(
            url, f.source.read_bytes(), _content_type_for(path)
        )
        if put_resp.status not in (200, 201, 204):
            raise PublishError(
                f"Upload failed for {path}: HTTP {put_resp.status}", file=path
            )

    notify("Committing…")
    commit_resp = transport.post_json(
        f"{base}/api/simulations/commit", {"upload_id": upload_id}, resolved_token
    )
    _raise_for_status(commit_resp, "commit")
    commit = commit_resp.json()
    sim_id = commit.get("id")
    if not sim_id:
        raise PublishError("commit response missing id.")

    return PublishResult(
        id=sim_id, sim_id=session.get("sim_id", sim_id), upload_id=upload_id
    )


def _default_title(config: dict) -> str:
    projects = config.get("projects") or []
    if projects and projects[0].get("name"):
        return str(projects[0]["name"])
    return "Untitled simulation"


def _raise_for_status(resp: HttpResponse, step: str) -> None:
    if resp.status in (200, 201):
        return
    # 422 responses are {error, file?}; surface them verbatim.
    try:
        payload = resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}
    error = payload.get("error") or f"{step} failed with HTTP {resp.status}"
    raise PublishError(error, file=payload.get("file"))


__all__ = [
    "DATA_EXTENSIONS",
    "DEFAULT_API_BASE",
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "MAX_TOTAL_BYTES",
    "TOKEN_ENV_VAR",
    "HttpResponse",
    "HttpTransport",
    "PublishError",
    "PublishPlan",
    "PublishResult",
    "plan_publish",
    "publish_dist",
    "resolve_token",
]
