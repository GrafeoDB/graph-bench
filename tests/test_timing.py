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


class _Clock:
    """One fake clock behind both `time.perf_counter_ns` and `time.monotonic`: only `advance` moves it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ns = 0
        monkeypatch.setattr(timing.time, "perf_counter_ns", lambda: self.ns)
        monkeypatch.setattr(timing.time, "monotonic", lambda: self.ns / 1e9)

    def advance(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)


def _runs_of(clock: _Clock, seconds: float) -> Any:
    calls: list[int] = []

    def fn() -> int:
        clock.advance(seconds)
        calls.append(1)
        return len(calls)

    return fn


def test_case_stops_starting_runs_after_the_time_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """No new timed run once the timed total reached the box and the minimum count is done; fast cases keep all."""
    clock = _Clock(monkeypatch)
    slow, _ = measure(_runs_of(clock, 2.0), warmup=2, iterations=10, time_box_s=5.0, min_iterations=3)
    assert slow.iterations == 3  # 2 + 2 + 2 >= 5 s after the third run; warm-ups do not count
    longer, _ = measure(_runs_of(clock, 10.0), warmup=0, iterations=10, time_box_s=5.0, min_iterations=3)
    assert longer.iterations == 3  # the minimum, though the first run alone filled the box
    fast, _ = measure(_runs_of(clock, 0.1), warmup=1, iterations=10, time_box_s=5.0, min_iterations=3)
    assert fast.iterations == 10
    load, _ = measure(_runs_of(clock, 10.0), warmup=0, iterations=3, time_box_s=5.0, min_iterations=1)
    assert load.iterations == 1
    unboxed, _ = measure(_runs_of(clock, 2.0), warmup=0, iterations=4)
    assert unboxed.iterations == 4


def test_measure_stops_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No run starts at or after the deadline, nor one the previous run's duration says would end past `kill_at`;
    with the minimum done the timing stands over the runs that ran, else `OverCeiling`."""
    clock = _Clock(monkeypatch)
    timed, _ = measure(_runs_of(clock, 10.0), warmup=0, iterations=10, min_iterations=3, deadline=35.0)
    assert timed.iterations == 4  # the 4th run starts at 30 s, before the deadline
    clock.ns = 0
    timed, _ = measure(_runs_of(clock, 10.0), warmup=0, iterations=10, min_iterations=3, deadline=35.0, kill_at=38.0)
    assert timed.iterations == 3  # a 4th would end at 40 s, past the kill
    clock.ns = 0
    with pytest.raises(timing.OverCeiling) as over:
        measure(_runs_of(clock, 10.0), warmup=2, iterations=10, min_iterations=3, deadline=35.0)
    assert over.value.runs == 2  # warm-ups at 0 and 10 s, timed runs at 20 and 30 s; none at 40 s
    clock.ns = 0
    with pytest.raises(timing.OverCeiling) as over:
        measure(_runs_of(clock, 10.0), warmup=5, iterations=10, min_iterations=3, deadline=35.0)
    assert over.value.runs == 0  # stopped during the warm-ups


def test_measure_predicts_the_first_run_from_the_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """`first_s`, the preflight's duration, predicts the first run: a single 40 s call whose preflight ended at
    40 s cannot end before the kill at 65 s, so it is not started (`OverCeiling(0)`, no call), where without the
    prediction it would start and be killed. One that fits (a 10 s call) still runs."""
    clock = _Clock(monkeypatch)
    clock.advance(40.0)  # the preflight's single call
    fn = _runs_of(clock, 40.0)
    with pytest.raises(timing.OverCeiling) as over:
        measure(fn, warmup=0, iterations=3, min_iterations=3, deadline=60.0, kill_at=65.0, first_s=40.0)
    assert over.value.runs == 0
    assert clock.ns == 40 * 10**9  # never called
    unseeded = _runs_of(clock, 40.0)
    with pytest.raises(timing.OverCeiling) as over:
        measure(unseeded, warmup=0, iterations=3, min_iterations=3, deadline=60.0, kill_at=65.0)
    assert over.value.runs == 1  # started at 40 s and ran to 80 s: the run the supervisor would have killed
    clock.ns = 40 * 10**9
    fits, _ = measure(_runs_of(clock, 5.0), warmup=0, iterations=3, deadline=60.0, kill_at=65.0, first_s=5.0)
    assert fits.iterations == 3
