r"""
Tests for graph_bench.adapters module.
"""

from typing import Any, ClassVar

import networkx as nx
import pytest

from graph_bench.adapters import AdapterRegistry, BaseAdapter

from .dict_adapter import DictAdapter


class TestAdapterRegistry:
    def test_registry_has_adapters(self):
        adapters = AdapterRegistry.list()
        assert "neo4j" in adapters
        assert "memgraph" in adapters
        assert "arangodb" in adapters
        assert "grafeo" in adapters
        assert "ladybug" in adapters
        assert "tugraph" in adapters
        assert "falkordblite" in adapters

    def test_get_adapter_class(self):
        adapter_cls = AdapterRegistry.get("neo4j")
        assert adapter_cls is not None
        assert issubclass(adapter_cls, BaseAdapter)

    def test_get_unknown_adapter(self):
        adapter_cls = AdapterRegistry.get("unknown")
        assert adapter_cls is None

    def test_create_adapter_unknown(self):
        with pytest.raises(ValueError, match="Unknown adapter"):
            AdapterRegistry.create("unknown")


class TestBaseAdapter:
    def test_base_adapter_is_abstract(self):
        abstract: type[Any] = BaseAdapter
        with pytest.raises(TypeError):
            abstract()

    def test_adapter_repr(self):
        from graph_bench.adapters.neo4j import Neo4jAdapter

        adapter = Neo4jAdapter()
        assert "Neo4j" in repr(adapter)
        assert "disconnected" in repr(adapter)


class TestNeo4jAdapter:
    def test_create_adapter(self):
        from graph_bench.adapters.neo4j import Neo4jAdapter

        adapter = Neo4jAdapter()
        assert adapter.name == "Neo4j"
        assert adapter.connected is False

    def test_adapter_version_disconnected(self):
        from graph_bench.adapters.neo4j import Neo4jAdapter

        adapter = Neo4jAdapter()
        assert adapter.version == "unknown"


class TestMemgraphAdapter:
    def test_create_adapter(self):
        from graph_bench.adapters.memgraph import MemgraphAdapter

        adapter = MemgraphAdapter()
        assert adapter.name == "Memgraph"
        assert adapter.connected is False


class TestArangoDBAdapter:
    def test_create_adapter(self):
        from graph_bench.adapters.arangodb import ArangoDBAdapter

        adapter = ArangoDBAdapter()
        assert adapter.name == "ArangoDB"
        assert adapter.connected is False


class TestTuGraphAdapter:
    def test_create_adapter(self):
        from graph_bench.adapters.tugraph import TuGraphAdapter

        adapter = TuGraphAdapter()
        assert adapter.name == "TuGraph"
        assert adapter.connected is False

    def test_adapter_version_disconnected(self):
        from graph_bench.adapters.tugraph import TuGraphAdapter

        adapter = TuGraphAdapter()
        assert adapter.version == "unknown"


class TestFalkorDBLiteAdapter:
    def test_create_adapter(self):
        from graph_bench.adapters.falkordblite import FalkorDBLiteAdapter

        adapter = FalkorDBLiteAdapter()
        assert adapter.name == "FalkorDB Lite"
        assert adapter.connected is False
        assert adapter.is_embedded is True

    def test_adapter_version_disconnected(self):
        from graph_bench.adapters.falkordblite import FalkorDBLiteAdapter

        adapter = FalkorDBLiteAdapter()
        assert adapter.version == "unknown"


class TestGrafeoAdapter:
    def test_create_adapter(self):
        from graph_bench.adapters.grafeo import GrafeoAdapter

        adapter = GrafeoAdapter()
        assert adapter.name == "Grafeo"
        assert adapter.connected is False


def _pagerank_reference(edges: list[tuple[str, str]], damping: float, iterations: int) -> dict[str, float]:
    """Exactly `iterations` power-iteration steps, sink rank spread evenly (LDBC Graphalytics)."""
    nodes = sorted({n for edge in edges for n in edge})
    out = {n: [t for s, t in edges if s == n] for n in nodes}
    rank = dict.fromkeys(nodes, 1.0 / len(nodes))
    for _ in range(iterations):
        sinks = sum(rank[n] for n in nodes if not out[n])
        incoming = dict.fromkeys(nodes, 0.0)
        for n in nodes:
            for t in out[n]:
                incoming[t] += rank[n] / len(out[n])
        rank = {n: (1 - damping) / len(nodes) + damping * (incoming[n] + sinks / len(nodes)) for n in nodes}
    return rank


class TestBaseAdapterFallbacks:
    EDGES: ClassVar[list[tuple[str, str]]] = [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")]  # d is a sink

    def test_pagerank_fallback_runs(self):
        # Regression: networkx needs scipy for pagerank, which was not a dependency.
        scores = DictAdapter(self.EDGES).pagerank(damping=0.85, max_iterations=100, tolerance=1e-8)
        assert set(scores) == {"a", "b", "c", "d"}
        assert sum(scores.values()) == pytest.approx(1.0)

    def test_pagerank_fallback_converges_like_networkx(self):
        scores = DictAdapter(self.EDGES).pagerank(damping=0.85, max_iterations=1000, tolerance=1e-12)
        graph = nx.DiGraph(self.EDGES)
        assert scores == pytest.approx(nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000), abs=1e-9)

    @pytest.mark.parametrize("iterations", [1, 2, 20])
    def test_pagerank_fallback_stops_after_max_iterations(self, iterations):
        # Regression: LDBC PageRank asks for 20 iterations at tolerance 1e-10, and
        # nx.pagerank raised PowerIterationFailedConvergence instead of returning them.
        scores = DictAdapter(self.EDGES).pagerank(damping=0.85, max_iterations=iterations, tolerance=1e-10)
        assert scores == pytest.approx(_pagerank_reference(self.EDGES, 0.85, iterations), abs=1e-12)

    def test_ldbc_pagerank_benchmark_runs_on_the_fallback(self, tiny_scale):
        from graph_bench.benchmarks.graphanalytics import LdbcPageRankBenchmark

        metrics = LdbcPageRankBenchmark().run(DictAdapter(), tiny_scale)
        assert metrics.items_processed == tiny_scale.measurement_iterations * 100  # a score per vertex
