"""Custom termination registrations for the MuscleMimic Fullbody demo.

Import this module before calling ``builder.build()`` to register the
``mimic_deviation`` early-termination term.

Registration is by *term name* (dict key) rather than by function name because
the mjlab implementation uses a closure (named ``_fn``).  The mjswan adapter
falls back to the term name when the function-name lookup fails.
"""

from __future__ import annotations

from pathlib import Path

from mjswan import TerminationBinding, register_termination

_TERM_DIR = Path(__file__).resolve().parent

register_termination(
    "mimic_deviation",
    TerminationBinding(
        ts_name="MimicDeviation",
        ts_src=str(_TERM_DIR / "MimicDeviation.ts"),
    ),
)
