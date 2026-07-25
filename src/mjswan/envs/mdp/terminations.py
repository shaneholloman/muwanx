"""Custom termination registry for mjswan (ADR 0005).

mjswan no longer carries its own reimplementation of mjlab's termination
functions. A task's real function object — mjlab's own, or an author's plain
``func(env, **params) -> Tensor`` written against the same live-env API — is
traced directly to ONNX at build time (:mod:`mjswan.compile`); a term that
reads no time-varying state (e.g. mjlab's own ``time_out``) is classified as
native automatically. There is no mjswan-side mirror to resolve by name.

This module now only carries the ``TerminationBinding`` escape hatch: mark a
term as ``unsupported_reason`` (accepted for API compatibility but not
available in the browser, e.g. ``illegal_contact``/``nan_detection``) or
``ts_src`` (a hand-written TS class). Register overrides via
:func:`register_termination`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TerminationBinding:
    """Binding from an mjlab termination name to its browser implementation.

    See ADR 0003.  Declarative terminations are plain traceable callables
    passed to ``TerminationTermCfg(func=...)``; a binding is the mjlab-name
    resolution layer carrying a ``ts_src`` escape hatch or ``unsupported_reason``.

    Attributes:
        ts_name: The TypeScript termination class name in the
            ``Terminations`` registry (e.g. ``"TimeOut"``).
        defaults: Default parameters merged into the JSON config entry.
        unsupported_reason: If set, this sentinel is accepted for API
            compatibility but raises ``NotImplementedError`` at build time.
        ts_src: Absolute path to a ``.ts`` file that exports the class
            named ``ts_name``. When set, the file is injected into the
            browser bundle at build time so the custom termination class is
            available to the browser-side ``TerminationManager``. Leave
            ``None`` for built-in classes already present in
            ``terminations.ts``.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    unsupported_reason: str | None = None
    ts_src: str | None = None


# ---------------------------------------------------------------------------
# Custom termination registry
# ---------------------------------------------------------------------------

_custom_registry: dict[str, TerminationBinding] = {}
"""Maps mjlab termination function names to user-supplied bindings.

Populated via :func:`register_termination`. The mjlab adapter checks this
registry as a fallback after the built-in sentinel lookup fails.
"""


def register_termination(mjlab_name: str, sentinel: TerminationBinding) -> None:
    """Register a custom termination binding for an mjlab termination.

    Call this before :meth:`~mjswan.Builder.build` so the adapter can
    resolve the function and the builder can inject any custom TypeScript
    source into the browser bundle.

    Args:
        mjlab_name: The mjlab termination function name
            (e.g. ``"out_of_terrain_bounds"``).
        sentinel: A :class:`TerminationBinding` describing the browser-side
            implementation. Set ``unsupported_reason`` to mark the
            termination as unsupported. Set ``ts_src`` to the absolute path
            of a ``.ts`` file that exports the class named by ``ts_name``.
    """
    _custom_registry[mjlab_name] = sentinel


illegal_contact = TerminationBinding(
    ts_name="",
    unsupported_reason=(
        "illegal_contact is not supported in mjswan: contact force checks on "
        "specific bodies are not available in the browser runtime. "
        "This termination is a training-time safety check and is not needed "
        "for browser-side policy inference."
    ),
)
"""Terminate when a non-foot body makes illegal contact.

.. note::
    Not supported in mjswan. Accepted for API compatibility so that mjlab
    configs can be imported without modification, but raises
    ``NotImplementedError`` at build time.
"""

nan_detection = TerminationBinding(
    ts_name="",
    unsupported_reason=(
        "nan_detection is not supported in mjswan: NaN/Inf detection "
        "across the full physics state is not available in the browser runtime. "
        "The browser simulation will simply diverge visually if NaN occurs."
    ),
)
"""Terminate when NaN/Inf values appear in physics state.

.. note::
    Not supported in mjswan. Accepted for API compatibility so that mjlab
    configs can be imported without modification, but raises
    ``NotImplementedError`` at build time.
"""


__all__ = [
    "TerminationBinding",
    "register_termination",
    "_custom_registry",
    "illegal_contact",
    "nan_detection",
]
