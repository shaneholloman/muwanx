"""Event manager configuration for mjswan.

Provides ``EventTermCfg`` for scene-level reset events such as spawn
position randomization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..envs.mdp.events import EventFunc


@dataclass
class EventTermCfg:
    """Configuration for a single event term.

    Mirrors ``mjlab.managers.event_manager.EventTermCfg``.
    Only ``mode="reset"`` events are supported in the browser runtime.

    ``func`` accepts either:

    - A legacy :class:`EventFunc` sentinel: emits ``{"name": ..., "params": ...}``
      and the engine resolves the class from its registry.
    - A DSL builder ``func(env, **params) -> list[Mutation]``: the build traces
      it into a ``{"kind": "event", "mutations": [...]}`` envelope (ADR 0003).
    """

    func: EventFunc | Callable[..., Any]
    """Event function — EventFunc sentinel (legacy) or DSL mutation builder."""

    mode: str = "reset"
    """Event trigger mode. Only ``"reset"`` is handled by the browser runtime."""

    params: dict[str, Any] = field(default_factory=dict)
    """Parameters forwarded to the TS event constructor or DSL builder."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for the TS ``EventManager``."""
        if isinstance(self.func, EventFunc):
            entry: dict[str, Any] = {"name": self.func.ts_name}
            merged: dict[str, Any] = {**self.func.defaults, **self.params}
            if merged:
                entry["params"] = merged
            return entry
        from ..dsl import trace_event

        return trace_event(self.func, self.params)


__all__ = ["EventTermCfg"]
