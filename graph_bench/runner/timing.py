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


def measure(fn: Callable[[], Any], *, warmup: int, iterations: int) -> tuple[Timing, Any]:
    """Call `fn` `warmup` times untimed, then `iterations` times timed; return the timing and the last result.

    The garbage collector is disabled during each timed call and run between calls.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    result: Any = None
    for _ in range(warmup):
        result = fn()
    samples: list[int] = []
    was_enabled = gc.isenabled()
    try:
        for _ in range(iterations):
            gc.collect()
            gc.disable()
            start = time.perf_counter_ns()
            result = fn()
            samples.append(time.perf_counter_ns() - start)
            gc.enable()
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
    timing = Timing(float(statistics.median(samples)), float(_p99(samples)), min(samples), iterations)
    return timing, result
