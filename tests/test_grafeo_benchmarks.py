r"""
Every registered benchmark runs on Grafeo, measures real work and says how.

Each benchmark runs end to end (setup, warmup, measurement) at a tiny scale on
embedded Grafeo, and on grafeo-server when one answers. A benchmark passes only
when it finishes and processes items; where the generated data fixes the
answer, the item count must match it exactly, so a query that returns too
little (or nothing) is caught even when it does not raise. Each result must
also carry the implementation label the adapter is expected to produce.
"""

from __future__ import annotations

import pytest

from graph_bench.adapters.base import BaseAdapter
from graph_bench.adapters.grafeo import GrafeoAdapter
from graph_bench.benchmarks import BenchmarkRegistry
from graph_bench.types import ScaleConfig, UnsupportedOperation

TINY = ScaleConfig(
    name="tiny", nodes=200, edges=1000, warmup_iterations=1, measurement_iterations=2, timeout_seconds=60
)

# Items over the 2 measured iterations, derived from each benchmark's generator at TINY.
EXPECTED_ITEMS = {
    "node_insertion": 2 * 20,  # 20 nodes per batch
    "edge_insertion": 2 * 100,  # 100 edges between existing nodes
    "single_read": 2 * 100,  # 100 lookups of existing nodes
    "batch_read": 2 * 200,  # all 200 Person nodes
    "aggregation_count": 2 * 50 * 20,  # 50 counts of 20 Person nodes
    "filter_equality": 2 * 20,  # 20 cities with one person each
    "filter_range": 2 * 165,  # ages 20..39, windows of 11 years
    "property_update": 2 * 50,  # 50 updates
    "graph_density": 2 * 11000,  # 33 edges between 6 nodes, scaled by 10000
    "ldbc_pagerank": 2 * 100,  # a score per vertex
    "ldbc_wcc": 2 * 100,  # every vertex in a component
    "ldbc_lcc": 2 * 100,  # a coefficient per vertex
    "lcc_undirected": 2 * 100,  # a coefficient per vertex
    "vector_scan_insert": 2 * 500,  # 500 vectors (the minimum)
    "vector_scan_knn": 2 * 10 * 10,  # 10 centroid queries, k=10
    "vector_scan_batch_search": 2 * 100 * 10,  # 100 queries, k=10
    "vector_ann_build": 2 * 500,
    "vector_ann_knn": 2 * 100 * 10,  # 100 queries, k=10
    "vector_ann_delete": 2 * 100 * 10,
}

# Zero items is a correct outcome here at TINY scale.
ZERO_ALLOWED = {
    "betweenness_centrality",  # 2 nodes: no path has an intermediate node
}

# What runs in Python on each Grafeo transport; everything else must be native.
FALLBACKS: dict[str, dict[str, set[str]]] = {
    "embedded": {
        "dfs": {"traverse_dfs"},
        "ldbc_lcc": {"local_clustering_coefficient"},  # Grafeo's LCC is undirected
        "snb_ic1": {"traverse_bfs"},  # BFS over one edge type
        "vector_scan_knn": {"vector_scan"},
        "vector_scan_batch_search": {"vector_scan"},
        "vector_scan_recall": {"vector_scan"},
        "hybrid_graph_to_vector": {"vector_scan"},
        "hybrid_vector_to_graph": {"vector_scan"},
    },
    "server": {
        "dfs": {"traverse_dfs"},
        "ldbc_lcc": {"local_clustering_coefficient"},
        "snb_ic1": {"traverse_bfs"},
        "shortest_path": {"shortest_path"},
        "betweenness_centrality": {"shortest_path"},
        "vector_scan_knn": {"vector_scan"},
        "vector_scan_batch_search": {"vector_scan"},
        "vector_scan_recall": {"vector_scan"},
        "hybrid_graph_to_vector": {"vector_scan"},
        "hybrid_vector_to_graph": {"vector_scan"},
    },
}


def benchmark_names() -> list[str]:
    return sorted(BenchmarkRegistry.list())


def is_unretryable_write_conflict(error: RuntimeError) -> bool:
    """The grafeo-server 0.5.44 bug: a write-write conflict comes back as 42001, not class 40.

    Concurrent writers to one node abort now and then; the adapter retries
    class-40 aborts, but this code says "syntax or access error".
    """
    from graph_bench.adapters.grafeo_server import GrafeoServerError

    return isinstance(error, GrafeoServerError) and error.code == "42001" and "Write conflict" in error.message


def test_expectations_name_real_benchmarks() -> None:
    names = set(benchmark_names())
    assert set(EXPECTED_ITEMS) | ZERO_ALLOWED <= names
    for fallbacks in FALLBACKS.values():
        assert set(fallbacks) <= names


@pytest.mark.parametrize("name", benchmark_names())
def test_benchmark_runs_on_grafeo(grafeo_adapter: BaseAdapter, name: str) -> None:
    bench_cls = BenchmarkRegistry.get(name)
    assert bench_cls is not None
    kind = "embedded" if isinstance(grafeo_adapter, GrafeoAdapter) else "server"
    try:
        metrics = bench_cls().run(grafeo_adapter, TINY)
    except UnsupportedOperation as unsupported:
        # Only a server built without the vector index may lack a native implementation.
        if kind == "embedded" or bench_cls.category != "vector_ann":
            raise
        pytest.skip(str(unsupported))
    except RuntimeError as error:
        if kind == "embedded" or not is_unretryable_write_conflict(error):
            raise
        pytest.skip(f"grafeo-server reports a write conflict as 42001, so it cannot be retried: {error}")

    assert metrics.timing.iterations == TINY.measurement_iterations
    if name in EXPECTED_ITEMS:
        assert metrics.items_processed == EXPECTED_ITEMS[name]
    elif bench_cls.category != "ldbc_acid" and name not in ZERO_ALLOWED:
        # ACID tests count passed checks; the others must have done some work.
        assert metrics.items_processed > 0

    expected = FALLBACKS[kind].get(name)
    if expected:
        assert metrics.custom["implementation"] == "fallback"
        assert set(metrics.custom["fallbacks"]) == expected
    else:
        assert metrics.custom["implementation"] == "native", metrics.custom.get("fallbacks")
