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


class OverCeiling(Exception):
    """`measure` stopped at its deadline with fewer timed runs than its minimum (`runs`): the case is too slow."""

    def __init__(self, runs: int) -> None:
        super().__init__(f"stopped at the ceiling after {runs} timed runs")
        self.runs = runs


def measure(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    on_run: Callable[[], None] | None = None,
    before: Callable[[], None] | None = None,
    after: Callable[[Any], None] | None = None,
    time_box_s: float | None = None,
    min_iterations: int = 1,
    deadline: float | None = None,
    kill_at: float | None = None,
    first_s: float | None = None,
) -> tuple[Timing, Any]:
    """Call `fn` `warmup` times untimed, then `iterations` times timed; return the timing and the last result.

    The garbage collector is disabled during each timed call and run between calls. `before`, if given, is called
    before every call and `after(result)` after it (warm-up and timed alike), both untimed and outside the window
    in which the collector is disabled: a write case's setup, and its read-back and teardown. `on_run`, if given,
    is called after every call (after `after`), outside the timed region too (the worker reports progress with it).

    Limits (Task T), each optional; `min_iterations` is capped at `iterations`:
    - `time_box_s`: no new timed run once the timed runs total that many seconds and `min_iterations` ran;
    - `deadline` (a `time.monotonic()` value): no warm-up or timed run starts at or after it, nor one that, as long
      as the previous run with its `before` and `after`, would end past `kill_at` (where a supervisor kills it).
      `first_s` (the preflight's duration, in seconds) stands in for the previous run before the first one, so a
      single call that cannot end before `kill_at` is not started. Stopped there with at least `min_iterations`
      timed runs, the timing is over those; with fewer it raises `OverCeiling`.
    The timing's `iterations` is the number of timed runs that ran.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    needed = min(max(min_iterations, 1), iterations)
    # The wall time of the previous run, `before` and `after` included; before the first, the preflight's.
    last: list[float] = [] if first_s is None else [first_s]

    def may_start() -> bool:
        if deadline is None:
            return True
        now = time.monotonic()
        if now >= deadline:
            return False
        return not (last and kill_at is not None and now + last[0] > kill_at)

    def clocked(run: Callable[[], Any]) -> Any:
        begun = time.monotonic()
        out = run()
        last[:] = [time.monotonic() - begun]
        return out

    def warm() -> Any:
        if before is not None:
            before()
        out = fn()
        if after is not None:
            after(out)
        return out

    result: Any = None
    for _ in range(warmup):
        if not may_start():
            raise OverCeiling(0)
        result = clocked(warm)
        if on_run is not None:
            on_run()
    samples: list[int] = []
    was_enabled = gc.isenabled()

    def timed() -> Any:
        if before is not None:
            before()
        gc.collect()
        gc.disable()
        start = time.perf_counter_ns()
        out = fn()
        samples.append(time.perf_counter_ns() - start)
        gc.enable()
        if after is not None:
            after(out)
        return out

    try:
        for _ in range(iterations):
            if time_box_s is not None and len(samples) >= needed and sum(samples) >= time_box_s * 1e9:
                break
            if not may_start():
                if len(samples) >= needed:
                    break
                raise OverCeiling(len(samples))
            result = clocked(timed)
            if on_run is not None:
                on_run()
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
    return summary(samples), result
