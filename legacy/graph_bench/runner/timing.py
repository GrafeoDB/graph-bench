r"""
Timing utilities for benchmarks.

    from graph_bench.runner.timing import Timer, measure_time
"""

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

__all__ = ["Timer", "measure_time", "measure_iterations"]

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class TimerResult:
    """Result from a timing measurement.

    Attributes:
        elapsed_ns: Elapsed time in nanoseconds.
        result: Return value from the timed function.
    """

    elapsed_ns: int
    result: Any = None

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000


class Timer:
    """Context manager for timing code blocks.

        with Timer() as t:
            do_something()
        print(f"Elapsed: {t.elapsed_ms}ms")
    """

    def __init__(self) -> None:
        self._start: int = 0
        self._end: int = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._end = time.perf_counter_ns()

    @property
    def elapsed_ns(self) -> int:
        """Elapsed time in nanoseconds."""
        return self._end - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000


def measure_time(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> TimerResult:
    """Measure execution time of a function call.

    Args:
        func: Function to call.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        TimerResult with elapsed time and function result.
    """
    start = time.perf_counter_ns()
    result = func(*args, **kwargs)
    end = time.perf_counter_ns()
    return TimerResult(elapsed_ns=end - start, result=result)


def measure_iterations(
    func: Callable[[], Any],
    *,
    iterations: int = 10,
    warmup: int = 3,
) -> list[int]:
    """Measure execution time over multiple iterations.

    Args:
        func: Function to call (no arguments).
        iterations: Number of measurement iterations.
        warmup: Number of warmup iterations (not counted).

    Returns:
        List of elapsed times in nanoseconds.
    """
    for _ in range(warmup):
        func()

    timings = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        timings.append(end - start)

    return timings


@contextmanager
def timed_section(name: str, *, callback: Callable[[str, int], None] | None = None):
    """Context manager for timing named code sections.

    Args:
        name: Name of the section being timed.
        callback: Optional callback(name, elapsed_ns) called on exit.

    Yields:
        Timer instance.
    """
    timer = Timer()
    timer.__enter__()
    try:
        yield timer
    finally:
        timer.__exit__(None, None, None)
        if callback:
            callback(name, timer.elapsed_ns)
