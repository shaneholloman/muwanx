"""Custom event registry for mjswan (ADR 0005).

mjswan no longer carries its own reimplementation of mjlab's event functions.
A task's real function object — mjlab's own, or an author's plain
``func(env, env_ids, **params) -> None`` written against the same live-env
API (writing via ``entity.write_*_to_sim``) — is traced directly to ONNX at
build time (:mod:`mjswan.compile`); a term that writes nothing traceable
(e.g. mjlab's own ``randomize_terrain``, which only mutates the terrain
generator) is classified as native automatically. There is no mjswan-side
mirror to resolve by name.

This module now only carries the ``EventBinding`` escape hatch (``ts_src`` /
``defaults``) for events still backed by a hand-written TS class. Register
overrides via :func:`register_event`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EventBinding:
    """Binding from an mjlab event name to its browser implementation.

    See ADR 0003.  Declarative reset events are plain builders returning
    ``list[Mutation]``; a binding is the mjlab-name resolution layer for
    events still backed by an engine class (``ts_src`` or a built-in name).

    Attributes:
        ts_name: The TypeScript event class name in the ``Events`` registry.
        defaults: Default parameters merged into the JSON config entry.
        ts_src: Absolute path to a ``.ts`` file that exports the class
            named ``ts_name``. When set, the file is injected into the
            browser bundle at build time. Leave ``None`` for built-in classes.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    ts_src: str | None = None


_custom_registry: dict[str, EventBinding] = {}


def register_event(mjlab_name: str, sentinel: EventBinding) -> None:
    """Register a custom event binding for an mjlab event function."""
    _custom_registry[mjlab_name] = sentinel


__all__ = [
    "EventBinding",
    "register_event",
    "_custom_registry",
]
