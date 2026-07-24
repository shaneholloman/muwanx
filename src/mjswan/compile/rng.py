"""RNG spy/replay for the parity harness (ADR 0005 §2 companion brief §2b).

**Build/test-time only — never ships to the browser.** Unrelated to the
orchestrator-owned seeded PRNG that governs mjswan's *runtime* randomness and
its bit-for-bit session replay (ADR 0005 §2); do not conflate the two.

Event/Command term bodies take ``rand`` as an explicit ONNX input rather than
drawing their own randomness (ADR 0005 §2). So the parity harness cannot run
mjlab's reference rollout and the exported graph independently — they would draw
different numbers and diverge for reasons unrelated to whether the traced math is
correct. Instead the harness **records every draw mjlab actually makes** during
the reference computation, then **replays those exact values** into the graph's
``rand`` input.

This validates that the traced math reproduces mjlab's transformation of a given
draw into a final value. It does *not* validate that mjlab and mjswan draw the
same numbers at runtime — a separate, already-solved concern (ADR 0005 §2).

The spy patches the *term function's own module globals* (e.g.
``reset_joints_by_offset.__globals__["sample_uniform"]``), not the source module,
because mjlab imports the name at import time — patching the source would not
affect the already-bound reference.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

# mjlab RNG helpers a term may pull into its module namespace. The spy patches
# whichever of these names is actually present in the term's globals.
_RNG_NAMES = ("sample_uniform",)


class DrawRecorder:
    """Records the values a term's RNG calls return, in call order.

    Used as a context manager around a single term invocation on the live env:
    the wrapped RNG helpers still return mjlab's real draw (the reference rollout
    is unaffected), while each returned tensor is captured. ``rand_vector`` then
    concatenates them into the flat ``rand`` input to replay into the ONNX graph.
    """

    def __init__(self, func: Callable[..., Any]):
        self._globals = getattr(func, "__globals__", {})
        self._draws: list[torch.Tensor] = []
        self._saved: dict[str, Callable[..., Any]] = {}

    def __enter__(self) -> DrawRecorder:
        for name in _RNG_NAMES:
            real = self._globals.get(name)
            if real is None:
                continue
            self._saved[name] = real
            self._globals[name] = self._make_spy(real)
        return self

    def __exit__(self, *exc: object) -> None:
        for name, real in self._saved.items():
            self._globals[name] = real
        self._saved.clear()

    def _make_spy(self, real: Callable[..., Any]) -> Callable[..., Any]:
        def spy(*args: Any, **kwargs: Any) -> Any:
            out = real(*args, **kwargs)
            if isinstance(out, torch.Tensor):
                self._draws.append(out.detach().reshape(-1).clone())
            return out

        return spy

    @property
    def rand_dim(self) -> int:
        return int(sum(d.numel() for d in self._draws))

    @property
    def rand_vector(self) -> torch.Tensor:
        """Flat ``rand`` tensor: every draw concatenated in call order."""
        if not self._draws:
            return torch.zeros(0)
        return torch.cat(self._draws)


class ReplayRng:
    """Serves recorded draws back to a term as it runs, in call order.

    Installed in place of the term's RNG helpers *during tracing*: each call
    consumes the next ``numel(size)`` values from the ``rand`` input, reshaped to
    the requested size, ignoring the ``lower``/``upper`` bounds — the recorded
    value already lies in range (it is the sampler's *output*, not a [0,1) base).
    This is what makes ``rand`` an explicit graph input per ADR 0005 §2.
    """

    def __init__(self, func: Callable[..., Any], rand: torch.Tensor):
        self._globals = getattr(func, "__globals__", {})
        self._rand = rand.reshape(-1)
        self._offset = 0
        self._saved: dict[str, Callable[..., Any]] = {}

    def __enter__(self) -> ReplayRng:
        for name in _RNG_NAMES:
            if name in self._globals:
                self._saved[name] = self._globals[name]
                self._globals[name] = self._consume
        return self

    def __exit__(self, *exc: object) -> None:
        for name, real in self._saved.items():
            self._globals[name] = real
        self._saved.clear()

    def _consume(self, lower: Any, upper: Any, size: Any, *args: Any, **kwargs: Any):
        n = int(torch.Size(size).numel())
        values = self._rand[self._offset : self._offset + n]
        self._offset += n
        return values.reshape(size)
