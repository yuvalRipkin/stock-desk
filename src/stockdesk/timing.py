"""Timing breakdowns: retrieval code reports where time goes; the agent collects it per search.

Code being measured wraps a step in `timed("embed_ms")` or sets a flag with `mark(...)`. That
only records something inside a `recording()` block, so the same code runs untimed elsewhere.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current: ContextVar[dict | None] = ContextVar("timings", default=None)


@contextmanager
def recording() -> Iterator[dict]:
    """Collect the timings reported inside the block into the yielded dict."""
    timings: dict = {}
    token = _current.set(timings)
    try:
        yield timings
    finally:
        _current.reset(token)


@contextmanager
def timed(key: str) -> Iterator[None]:
    """Add the block's wall time in ms to `key` (summed if the step runs several times)."""
    timings = _current.get()
    start = time.perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start) * 1000


def mark(key: str, value: object = True) -> None:
    """Set a flag or value on the current recording, if any."""
    timings = _current.get()
    if timings is not None:
        timings[key] = value


def elapsed_ms(start: float) -> float:
    """Milliseconds since a time.perf_counter() reading."""
    return (time.perf_counter() - start) * 1000
