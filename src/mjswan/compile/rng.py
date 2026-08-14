"""Build-time RNG spy/replay, so a traced graph can take ``rand`` as an explicit input.

Nothing here is the runtime's seeded PRNG. Recording mjlab's real draws and replaying
those exact values into the graph is what lets the parity harness compare the two
without them diverging on randomness alone.

The spy patches the term function's *own* module globals, since mjlab binds the name
at import time and patching the source module would not reach it.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

# mjlab RNG helpers a term may import; the spy patches whichever are in its globals.
_RNG_NAMES = ("sample_uniform",)


class DrawRecorder:
    """Records the values a term's RNG calls return, in call order.

    Wraps a single term invocation on the live env: the helpers still return mjlab's
    real draw, so the reference rollout is unaffected.
    """

    def __init__(self, func: Callable[..., Any]):
        self._globals = getattr(func, "__globals__", {})
        self._draws: list[torch.Tensor] = []
        self._bounds: list[torch.Tensor] = []
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
                self._bounds.append(_element_bounds(out.shape, *args, **kwargs))
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

    @property
    def rand_ranges(self) -> list[list[float]]:
        """Per-element ``[low, high]`` of the flat ``rand`` vector, in draw order.

        The graph consumes the sampler's *output* and so remembers no bounds, making
        these the only record of them — without which the runtime would draw [0, 1)
        and turn an empty ``pose_range`` into a metre of teleport per reset.
        """
        if not self._bounds:
            return []
        stacked = torch.cat(self._bounds)
        return [[float(low), float(high)] for low, high in stacked.tolist()]


def _element_bounds(shape: torch.Size, *args: Any, **kwargs: Any) -> torch.Tensor:
    """``(numel, 2)`` of the ``[low, high]`` behind each element of one draw.

    ``sample_uniform``'s bounds are scalars or broadcast against ``size``, so they are
    broadcast the same way here and flattened in the draw's own element order.
    """
    lower = kwargs.get("lower", args[0] if len(args) > 0 else 0.0)
    upper = kwargs.get("upper", args[1] if len(args) > 1 else 1.0)
    as_column = [
        torch.broadcast_to(torch.as_tensor(bound, dtype=torch.float32).cpu(), shape)
        .reshape(-1, 1)
        .clone()
        for bound in (lower, upper)
    ]
    return torch.cat(as_column, dim=1)


class ReplayRng:
    """Serves recorded draws back to a term as it runs, in call order.

    Installed in place of the term's RNG helpers during tracing, which is what turns
    ``rand`` into an explicit graph input. The bounds are ignored: a recorded value is
    the sampler's output, already in range.
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
