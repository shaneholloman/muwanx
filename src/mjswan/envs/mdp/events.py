"""Custom event registry for mjswan (ADR 0005).

mjswan carries no reimplementation of mjlab's event functions. A task's real
function object — mjlab's own, or an author's plain
``func(env, env_ids, **params) -> None`` written against the same live-env API
(writing via ``entity.write_*_to_sim``) — is traced directly to ONNX at build
time (:mod:`mjswan.compile`). There is no mjswan-side mirror to resolve by name.

This module carries only the ``EventBinding`` escape hatch: a hand-written TS
class for a term that cannot be expressed as a traced function at all. Register
one via :func:`register_event`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EventBinding:
    """A hand-written TS event class, bound to an mjlab event name.

    The escape hatch for a term ONNX tracing cannot express. Nothing else needs
    one: an authored term passes a traceable ``func=`` straight to
    ``EventTermCfg`` and is traced.

    Attributes:
        ts_name: Class the ``.ts`` file exports, and the name the browser's
            ``Events`` registry resolves.
        defaults: Default parameters merged into the JSON config entry.
        ts_src: Absolute path to the ``.ts`` file exporting ``ts_name``, injected
            into the browser bundle at build time. Without it there is no
            implementation to run and the build fails: mjswan ships no built-in
            TS event classes, since every built-in event is a traced graph or a
            model-field randomization the runtime draws itself.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    ts_src: str | None = None


_custom_registry: dict[str, EventBinding] = {}
"""Maps an mjlab event function name to its override.

Populated via :func:`register_event`; consulted by the mjlab adapter when the
config's own ``func`` needs replacing."""


def register_event(mjlab_name: str, sentinel: EventBinding) -> None:
    """Bind one mjlab event to a hand-written TS class.

    Call before :meth:`~mjswan.Builder.build`, so the adapter resolves the name
    and the builder injects ``ts_src`` into the browser bundle.

    Args:
        mjlab_name: The mjlab event function name (e.g. ``"push_robot"``).
        sentinel: An :class:`EventBinding` whose ``ts_src`` implements it.
    """
    _custom_registry[mjlab_name] = sentinel


__all__ = [
    "EventBinding",
    "register_event",
    "_custom_registry",
]
