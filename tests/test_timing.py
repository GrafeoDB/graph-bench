import gc

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
