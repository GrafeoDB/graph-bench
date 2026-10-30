r"""
Shared pytest fixtures for graph-bench tests.
"""

import functools
from collections.abc import Iterator

import pytest

from graph_bench.adapters.base import BaseAdapter
from graph_bench.config import SCALES
from graph_bench.types import ScaleConfig

# The grafeo-server tests use their own graph, so they never touch "bench".
GRAFEO_SERVER_TEST_GRAPH = "graph_bench_tests"


@functools.cache
def grafeo_server_unavailable() -> str | None:
    """Why the grafeo-server tests cannot run, or None when a GWP endpoint answers.

    Checked when a test needs the server: GRAPH_BENCH_GRAFEO_SERVER_URI, or
    localhost:7688 (grafeo-server's GWP port) when it is not set.
    """
    from graph_bench.adapters.grafeo_server import gwp_address

    address = gwp_address()
    try:
        import grpc
    except ImportError:
        return "grpcio is not installed (uv sync --extra grafeo-server)"
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=2)
    except grpc.FutureTimeoutError:
        return f"no grafeo-server GWP endpoint answers at {address} (set GRAPH_BENCH_GRAFEO_SERVER_URI)"
    finally:
        channel.close()
    return None


def connect_grafeo(kind: str) -> BaseAdapter:
    """Connect an empty Grafeo adapter: "embedded" (in memory) or "server" (GWP)."""
    if kind == "embedded":
        from graph_bench.adapters.grafeo import GrafeoAdapter

        adapter: BaseAdapter = GrafeoAdapter()
        adapter.connect(path=":memory:")
        return adapter
    if reason := grafeo_server_unavailable():
        pytest.skip(reason)
    from graph_bench.adapters.grafeo_server import GrafeoServerAdapter

    adapter = GrafeoServerAdapter(graph=GRAFEO_SERVER_TEST_GRAPH)
    adapter.connect()
    adapter.clear()
    return adapter


@pytest.fixture(params=["embedded", pytest.param("server", marks=pytest.mark.grafeo_server)])
def grafeo_adapter(request: pytest.FixtureRequest) -> Iterator[BaseAdapter]:
    """An empty Grafeo adapter, embedded and (when one answers) grafeo-server."""
    adapter = connect_grafeo(request.param)
    yield adapter
    adapter.disconnect()


@pytest.fixture
def small_scale() -> ScaleConfig:
    """Small scale configuration for testing."""
    return SCALES["small"]


@pytest.fixture
def tiny_scale() -> ScaleConfig:
    """Tiny scale for fast unit tests."""
    return ScaleConfig(
        name="tiny",
        nodes=100,
        edges=200,
        warmup_iterations=1,
        measurement_iterations=2,
        timeout_seconds=10,
    )


@pytest.fixture
def sample_nodes() -> list[dict]:
    """Sample node data for testing."""
    return [
        {"id": "person_0", "name": "Alice", "age": 30},
        {"id": "person_1", "name": "Bob", "age": 25},
        {"id": "person_2", "name": "Charlie", "age": 35},
        {"id": "person_3", "name": "Diana", "age": 28},
        {"id": "person_4", "name": "Eve", "age": 32},
    ]


@pytest.fixture
def sample_edges() -> list[tuple]:
    """Sample edge data for testing."""
    return [
        ("person_0", "person_1", "FOLLOWS", {"since": 2020}),
        ("person_0", "person_2", "FOLLOWS", {"since": 2021}),
        ("person_1", "person_2", "FOLLOWS", {"since": 2019}),
        ("person_2", "person_3", "FOLLOWS", {"since": 2022}),
        ("person_3", "person_4", "FOLLOWS", {"since": 2020}),
        ("person_4", "person_0", "FOLLOWS", {"since": 2021}),
    ]
