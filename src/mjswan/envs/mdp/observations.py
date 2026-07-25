"""Custom observation registry for mjswan (ADR 0005).

mjswan no longer carries its own reimplementation of mjlab's observation
functions. A task's real function object — mjlab's own, or an author's plain
``func(env, **params) -> Tensor`` written against the same live-env API — is
traced directly to ONNX at build time (:mod:`mjswan.compile`); there is no
mjswan-side mirror to resolve by name.

This module now only carries the ``ObservationBinding`` escape hatch: mark a
term as ``unsupported_reason`` (accepted for API compatibility but not
available in the browser, e.g. ``height_scan``) or ``ts_src`` (a hand-written
TS class, for a term that cannot be expressed as a traced function at all).
Register overrides via :func:`register_observation`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObservationBinding:
    """Binding from an mjlab observation name to its browser implementation.

    See ADR 0003.  A binding is the *mjlab-name resolution* layer; authors of
    new declarative terms pass a traceable ``func=`` callable straight to
    ``ObservationTermCfg`` and bypass this entirely.  A binding carries either
    a ``ts_src`` custom-JS escape hatch or an ``unsupported_reason`` marker;
    declarative built-ins are plain callables, not bindings.

    Attributes:
        ts_name: The TypeScript observation class name in the
            ``Observations`` registry (e.g. ``"BaseLinearVelocity"``).
        defaults: Default parameters merged into the JSON config entry.
            These map mjlab semantics to the existing TS class API.
        unsupported_reason: If set, this sentinel is accepted for API
            compatibility but raises ``NotImplementedError`` at build time
            with this message.
        ts_src: Absolute path to a ``.ts`` file that exports the class
            named ``ts_name``.  When set, the file is injected into the
            browser bundle at build time so the custom observation class is
            available to the ``PolicyRunner``.  Leave ``None`` for built-in
            classes already present in ``observations.ts``.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    unsupported_reason: str | None = None
    ts_src: str | None = None


# ---------------------------------------------------------------------------
# Custom observation registry
# ---------------------------------------------------------------------------

_custom_registry: dict[str, ObservationBinding | Callable[..., Any]] = {}
"""Maps mjlab observation function names to a user-supplied binding.

Populated via :func:`register_observation`.  The mjlab adapter checks this
registry as a fallback after the built-in lookup fails.  An entry is either
an :class:`ObservationBinding` (``ts_src`` escape hatch / unsupported marker)
or a **DSL builder callable** ``func(env, **params)`` for a task-specific
declarative term (ADR 0003).
"""


def register_observation(
    mjlab_name: str, sentinel: ObservationBinding | Callable[..., Any]
) -> None:
    """Register a custom observation binding for an mjlab observation function.

    Call this before :meth:`~mjswan.Builder.build` so the adapter can resolve
    the function.  ``sentinel`` may be:

    - a **DSL builder callable** ``func(env, **params)`` — the build traces it
      into a composition graph (declarative, no ``ts_src``); this is how
      task-specific terms (e.g. a task's ``ee_to_object_distance``) stay out of
      the core library while remaining Cloud-safe.
    - an :class:`ObservationBinding` with ``ts_src`` (custom-JS escape hatch) or
      ``unsupported_reason`` (accepted-but-unsupported marker).

    Args:
        mjlab_name: The mjlab observation function name
            (e.g. ``"ee_to_object_distance"``).
        sentinel: An :class:`ObservationBinding` describing the browser-side
            implementation.  Set ``unsupported_reason`` to mark the
            observation as unsupported (silently skipped at build time).
            Set ``ts_src`` to the absolute path of a ``.ts`` file that
            exports the class named by ``ts_name``.

    Example — mark as unsupported::

        register_observation(
            "ee_to_object_distance",
            ObservationBinding(ts_name="", unsupported_reason="not available in browser"),
        )

    Example — provide a custom TypeScript implementation::

        register_observation(
            "my_custom_obs",
            ObservationBinding(ts_name="MyCustomObs", ts_src="/path/to/MyCustomObs.ts"),
        )
    """
    _custom_registry[mjlab_name] = sentinel


height_scan = ObservationBinding(
    ts_name="",
    unsupported_reason=(
        "height_scan is not supported in mjswan: RayCastSensor is not "
        "available in the browser runtime."
    ),
)
"""Height scan from a RayCastSensor.

.. note::
    Not supported in mjswan. Accepted for API compatibility so that mjlab
    configs can be imported without modification, but raises
    ``NotImplementedError`` at build time.
"""


__all__ = [
    "ObservationBinding",
    "register_observation",
    "_custom_registry",
    "height_scan",
]
