"""mjlab-native ONNX compiler (ADR 0005).

Build-time tracing of mjlab MDP term bodies to ONNX graphs, plus a numeric
parity harness that validates the exported graphs against the live mjlab env.
Phase 1 covers value-returning terms (observations, non-native terminations).
"""

from __future__ import annotations

from .parity import ParityReport, TermReport, run_parity
from .rng import DrawRecorder, ReplayRng
from .tracer import EventExport, TermExport, trace_event_term, trace_term

__all__ = [
    "TermExport",
    "EventExport",
    "trace_term",
    "trace_event_term",
    "run_parity",
    "ParityReport",
    "TermReport",
    "DrawRecorder",
    "ReplayRng",
]
