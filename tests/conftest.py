from collections.abc import Iterator
from pathlib import Path

import pytest

from graph_bench.datasets import cache


@pytest.fixture(scope="session")
def push_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A `GRAPH_BENCH_DATA_DIR` holding the push selection's datasets and references, generated once per session."""
    root = tmp_path_factory.mktemp("push-data")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAPH_BENCH_DATA_DIR", str(root))
        cache.prepare("push")
    yield root
