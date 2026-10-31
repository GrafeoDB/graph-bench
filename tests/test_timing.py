import gc
from typing import Any

import pytest

from graph_bench.runner import timing
from graph_bench.runner.timing import measure


def test_measure_counts_iterations() -> None:
    calls: list[bool] = []

    def fn() -> int:
        calls.append(gc.isenabled())
        return len(calls)

    timing, result = measure(fn, warmup=2, iterations=5)
    assert timing.iterations == 5
    assert timing.min_ns <= timing.median_ns <= timing.p99_ns
    assert result == 7
    assert calls[2:] == [False] * 5
    assert gc.isenabled()


def test_measure_reports_each_run() -> None:
    reports: list[bool] = []
    measure(lambda: None, warmup=2, iterations=3, on_run=lambda: reports.append(gc.isenabled()))
    assert reports == [True] * 5


def test_measure_runs_before_and_after_outside_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake clock that only `before` and `after` advance: every sample is 0 ns, and they wrap every call."""
    clock = [0]
    monkeypatch.setattr(timing.time, "perf_counter_ns", lambda: clock[0])
    calls: list[str] = []

    def before() -> None:
        assert gc.isenabled()
        calls.append("before")
        clock[0] += 1000

    def after(result: Any) -> None:
        assert gc.isenabled()
        calls.append(f"after {result}")
        clock[0] += 1000

    def fn() -> int:
        calls.append("fn")
        return len(calls)

    result, last = measure(fn, warmup=1, iterations=2, before=before, after=after)
    assert (result.median_ns, result.p99_ns, result.min_ns, result.iterations) == (0.0, 0.0, 0, 2)
    assert calls == ["before", "fn", "after 2", "before", "fn", "after 5", "before", "fn", "after 8"]
    assert last == 8
