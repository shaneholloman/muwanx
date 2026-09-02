"""The document format number a build stamps, and what a reader owes it.

``format`` describes the *structure* of a build's output: what files exist, where they
sit, and what the manifest says about them. It is bumped by hand, only when an engine
reading the old structure would misread the new one. It is not the mjswan version,
which is stamped separately as ``version`` and says which release wrote the document
(ADR 0006 §7): a host picks an engine by ``version``; an engine protects itself by
``format``.

Absent means the layout that predates ADR 0006 (a root ``assets/config.json`` plus one
JSON per policy).
"""

from __future__ import annotations

#: The format this release writes. Bump only for a structural break.
DOCUMENT_FORMAT = 1

__all__ = ["DOCUMENT_FORMAT"]
