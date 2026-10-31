"""A pytest plugin for the benches tests (`-p tests.benchmark_spy`): records every use of the benchmark fixture.

Each call of `benchmark(...)` or `benchmark.pedantic(...)` appends the test's node id to the file named by
`GRAPH_BENCH_SPY`. With `GRAPH_BENCH_SPY_CALLS` set to a file, each call also appends `<node id> call` or
`<node id> pedantic setup=<bool> rounds=<n>` to it.
"""

import os
from typing import Any

from pytest_codspeed.plugin import BenchmarkFixture

_call = BenchmarkFixture.__call__
_pedantic = BenchmarkFixture.pedantic


def _record(fixture: BenchmarkFixture, how: str) -> None:
    nodeid = fixture._request.node.nodeid
    with open(os.environ["GRAPH_BENCH_SPY"], "a", encoding="utf-8") as fh:
        fh.write(nodeid + "\n")
    calls = os.environ.get("GRAPH_BENCH_SPY_CALLS")
    if calls:
        with open(calls, "a", encoding="utf-8") as fh:
            fh.write(f"{nodeid} {how}\n")


def _spy_call(self: BenchmarkFixture, *args: Any, **kwargs: Any) -> Any:
    _record(self, "call")
    return _call(self, *args, **kwargs)


def _spy_pedantic(self: BenchmarkFixture, *args: Any, **kwargs: Any) -> Any:
    _record(self, f"pedantic setup={kwargs.get('setup') is not None} rounds={kwargs.get('rounds', 1)}")
    return _pedantic(self, *args, **kwargs)


BenchmarkFixture.__call__ = _spy_call  # type: ignore[method-assign]
BenchmarkFixture.pedantic = _spy_pedantic  # type: ignore[method-assign]
