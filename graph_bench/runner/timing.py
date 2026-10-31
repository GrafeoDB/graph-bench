"""Wall-clock measurement of a repeated call."""

import gc
import statistics
import time
from collections.abc import Callable
from typing import Any

from graph_bench.types import Timing


def _p99(samples: list[int]) -> float:
    if len(samples) == 1:
        return float(samples[0])
    return statistics.quantiles(samples, n=100, method="inclusive")[98]


def summary(samples: list[int]) -> Timing:
    """The median, p99, minimum and count of `samples` (at least one), in nanoseconds."""
    return Timing(float(statistics.median(samples)), float(_p99(samples)), min(samples), len(samples))


def measure(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    on_run: Callable[[], None] | None = None,
    before: Callable[[], None] | None = None,
    after: Callable[[Any], None] | None = None,
) -> tuple[Timing, Any]:
    """Call `fn` `warmup` times untimed, then `iterations` times timed; return the timing and the last result.

    The garbage collector is disabled during each timed call and run between calls. `before`, if given, is called
    before every call and `after(result)` after it (warm-up and timed alike), both untimed and outside the window
    in which the collector is disabled: a write case's setup, and its read-back and teardown. `on_run`, if given,
    is called after every call (after `after`), outside the timed region too (the worker reports progress with it).
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    result: Any = None
    for _ in range(warmup):
        if before is not None:
            before()
        result = fn()
        if after is not None:
            after(result)
        if on_run is not None:
            on_run()
    samples: list[int] = []
    was_enabled = gc.isenabled()
    try:
        for _ in range(iterations):
            if before is not None:
                before()
            gc.collect()
            gc.disable()
            start = time.perf_counter_ns()
            result = fn()
            samples.append(time.perf_counter_ns() - start)
            gc.enable()
            if after is not None:
                after(result)
            if on_run is not None:
                on_run()
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
    return summary(samples), result
