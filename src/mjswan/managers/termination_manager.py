"""Termination manager configuration for mjswan.

Provides ``TerminationTermCfg`` with an API compatible with
``mjlab.managers.termination_manager``.

Example (identical to mjlab)::

    from mjlab.envs.mdp import terminations as term_fns
    from mjswan.managers.termination_manager import TerminationTermCfg

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

from ..envs.mdp.terminations import TerminationBinding


@dataclass
class TerminationTermCfg:
    """Configuration for a single termination term.

    Mirrors ``mjlab.managers.termination_manager.TerminationTermCfg``.

    ``func`` accepts either:

    - A legacy :class:`TerminationBinding` sentinel: the build emits the existing
      ``{"name": ..., "params": ...}`` shape and the engine resolves the
      class from its registry.
    - A plain Python callable taking ``(env, **params)``: the build traces
      it to ONNX against the scene's live env (:mod:`mjswan.compile`) — either
      mjlab's own function, or an author's function written against the same
      live-env API. A term that reads no time-varying state (e.g. mjlab's own
      ``time_out``) is classified as native automatically. See ADR 0005.
    """

    func: TerminationBinding | Callable[..., Any]
    """Termination function — TerminationBinding sentinel (legacy) or a
    plain callable traced to ONNX (ADR 0005)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to the function or TS constructor."""

    time_out: bool = False
    """Whether this term is a truncation (time-based) rather than a
    terminal failure.  Maps to the ``time_out`` flag in the JSON config."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize a **legacy** ``TerminationBinding`` term.

        Produces ``{"name": ..., "params": ..., "time_out": ...}`` for a term
        whose ``func`` is a ``TerminationBinding``.

        A term whose ``func`` is a plain callable is traced to ONNX against a
        live env at build time (ADR 0005), which this method has no access to.
        The Builder calls ``mjswan._onnx_build.serialize_termination`` for
        those instead; this method is not a valid way to serialize them.
        """
        if isinstance(self.func, TerminationBinding):
            return self._to_dict_legacy()
        raise TypeError(
            f"TerminationTermCfg.to_dict() cannot serialize a plain callable "
            f"func ({self.func!r}) — it must be traced to ONNX against a live "
            f"env. Use mjswan._onnx_build.serialize_termination(cfg, env, "
            f"out_dir) instead (the Builder does this automatically)."
        )

    def _to_dict_legacy(self) -> dict[str, Any]:
        func: TerminationBinding = self.func  # type: ignore[assignment]
        if func.unsupported_reason is not None:
            raise NotImplementedError(func.unsupported_reason)

        entry: dict[str, Any] = {"name": func.ts_name}
        merged: dict[str, Any] = {**func.defaults, **self.params}
        if merged:
            entry["params"] = merged
        if self.time_out:
            entry["time_out"] = True
        return entry


__all__ = ["TerminationTermCfg"]
