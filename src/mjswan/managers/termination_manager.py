"""Termination manager configuration for mjswan.

Provides ``TerminationTermCfg`` with an API compatible with
``mjlab.managers.termination_manager``.

Example (identical to mjlab)::

    from mjswan.managers.termination_manager import TerminationTermCfg
    from mjswan.envs.mdp import terminations as term_fns

    terminations = {
        "time_out": TerminationTermCfg(
            func=term_fns.time_out, time_out=True,
        ),
        "fallen": TerminationTermCfg(
            func=term_fns.bad_orientation,
            params={"limit_angle": 1.0},
        ),
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..envs.mdp.terminations import TermFunc


@dataclass
class TerminationTermCfg:
    """Configuration for a single termination term.

    Mirrors ``mjlab.managers.termination_manager.TerminationTermCfg``.

    ``func`` accepts either:

    - A legacy :class:`TermFunc` sentinel: the build emits the existing
      ``{"name": ..., "params": ...}`` shape and the engine resolves the
      class from its registry.
    - A plain Python callable taking ``(env, **params)``: the build traces
      the function against a symbolic env (see :mod:`mjswan.dsl`) and emits
      the composition graph instead.  This is the declarative path described
      in ADR 0003.
    """

    func: TermFunc | Callable[..., Any]
    """Termination function — TermFunc sentinel (legacy) or DSL callable."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to the function or TS constructor."""

    time_out: bool = False
    """Whether this term is a truncation (time-based) rather than a
    terminal failure.  Maps to the ``time_out`` flag in the JSON config."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for the TS ``TerminationManager``.

        Legacy ``TermFunc`` produces ``{"name": ..., "params": ..., "time_out": ...}``.
        A DSL callable produces ``{"kind": "termination", "nodes": [...], ...}``.
        """
        if isinstance(self.func, TermFunc):
            return self._to_dict_legacy()
        return self._to_dict_traced()

    def _to_dict_legacy(self) -> dict[str, Any]:
        func: TermFunc = self.func  # type: ignore[assignment]
        if func.unsupported_reason is not None:
            raise NotImplementedError(func.unsupported_reason)

        entry: dict[str, Any] = {"name": func.ts_name}
        merged: dict[str, Any] = {**func.defaults, **self.params}
        if merged:
            entry["params"] = merged
        if self.time_out:
            entry["time_out"] = True
        return entry

    def _to_dict_traced(self) -> dict[str, Any]:
        from ..dsl import trace_termination

        entry = trace_termination(self.func, self.params)  # type: ignore[arg-type]
        if self.time_out:
            entry["time_out"] = True
        return entry


__all__ = ["TerminationTermCfg"]
