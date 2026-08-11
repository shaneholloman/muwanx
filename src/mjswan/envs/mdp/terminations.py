"""Custom termination registry for mjswan (ADR 0005).

mjswan carries no reimplementation of mjlab's termination functions. A task's
real function object — mjlab's own, or an author's plain
``func(env, **params) -> Tensor`` written against the same live-env API — is
traced directly to ONNX at build time (:mod:`mjswan.compile`); a term that reads
no time-varying state (e.g. mjlab's own ``time_out``) is classified as native
automatically. There is no mjswan-side mirror to resolve by name.

This module carries only the ``TerminationBinding`` escape hatch: a hand-written
TS class for a term that cannot be expressed as a traced function at all.
Register one via :func:`register_termination`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TerminationBinding:
    """A hand-written TS termination class, bound to an mjlab termination name.

    The escape hatch for a term ONNX tracing cannot express. Nothing else needs
    one: an authored term passes a traceable ``func=`` straight to
    ``TerminationTermCfg`` and is traced.

    Attributes:
        ts_name: Class the ``.ts`` file exports, and the name the browser's
            ``Terminations`` registry resolves.
        defaults: Default parameters merged into the JSON config entry.
        ts_src: Absolute path to the ``.ts`` file exporting ``ts_name``, injected
            into the browser bundle at build time. Without it there is no
            implementation to run and the build fails: mjswan ships no built-in
            TS termination classes, since every built-in term is a traced graph
            or the native ``time_out``.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    ts_src: str | None = None


_custom_registry: dict[str, TerminationBinding] = {}
"""Maps an mjlab termination function name to its override.

Populated via :func:`register_termination`; consulted by the mjlab adapter when
the config's own ``func`` needs replacing."""


def register_termination(mjlab_name: str, sentinel: TerminationBinding) -> None:
    """Bind one mjlab termination to a hand-written TS class.

    Call before :meth:`~mjswan.Builder.build`, so the adapter resolves the name
    and the builder injects ``ts_src`` into the browser bundle.

    Args:
        mjlab_name: The mjlab termination function name (e.g. ``"illegal_contact"``).
        sentinel: A :class:`TerminationBinding` whose ``ts_src`` implements it.
    """
    _custom_registry[mjlab_name] = sentinel


__all__ = [
    "TerminationBinding",
    "register_termination",
    "_custom_registry",
]
