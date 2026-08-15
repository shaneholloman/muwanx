"""Custom observation registry.

mjswan reimplements none of mjlab's observation functions: a task's real function
object is traced to ONNX at build time. This module carries only the
``ObservationBinding`` escape hatch, for a term that cannot be traced at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObservationBinding:
    """A hand-written TS observation class, bound to an mjlab observation name.

    Attributes:
        ts_name: Class the ``.ts`` file exports, and the name the browser's
            ``Observations`` registry resolves.
        defaults: Default parameters merged into the JSON config entry.
        ts_src: Absolute path to the ``.ts`` file exporting ``ts_name``, injected into
            the bundle at build time. Required — mjswan ships no built-in TS classes,
            so without it the build fails.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    ts_src: str | None = None


_custom_registry: dict[str, ObservationBinding | Callable[..., Any]] = {}
"""Maps an mjlab observation function name to its override.

Populated via :func:`register_observation`; consulted by the mjlab adapter when
the config's own ``func`` needs replacing."""


def register_observation(
    mjlab_name: str, sentinel: ObservationBinding | Callable[..., Any]
) -> None:
    """Override how one mjlab observation function is exported.

    Call before :meth:`~mjswan.Builder.build`. ``sentinel`` is either a traceable
    ``func(env, **params) -> Tensor`` to trace in place of the task's own, or an
    :class:`ObservationBinding` naming a hand-written TS class.

    Args:
        mjlab_name: The mjlab observation function name (e.g. ``"height_scan"``).
        sentinel: A traceable callable, or an :class:`ObservationBinding`.

    Example::

        register_observation(
            "my_custom_obs",
            ObservationBinding(ts_name="MyCustomObs", ts_src="/path/to/MyCustomObs.ts"),
        )
    """
    _custom_registry[mjlab_name] = sentinel


__all__ = [
    "ObservationBinding",
    "register_observation",
    "_custom_registry",
]
