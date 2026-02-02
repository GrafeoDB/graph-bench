r"""
Tests for graph_bench.adapters module.
"""

import pytest

from graph_bench.adapters import AdapterRegistry, BaseAdapter


class TestAdapterRegistry:
    def test_registry_has_adapters(self):
        adapters = AdapterRegistry.list()
        assert "neo4j" in adapters
        assert "memgraph" in adapters
        assert "kuzu" in adapters
        assert "arangodb" in adapters
        assert "grafeo" in adapters

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
        with pytest.raises(TypeError):
            BaseAdapter()  # type: ignore

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


class TestKuzuAdapter:
    def test_create_adapter(self):
        from graph_bench.adapters.kuzu import KuzuAdapter

        adapter = KuzuAdapter()
        assert adapter.name == "Kuzu"
        assert adapter.connected is False


class TestArangoDBAdapter:
    def test_create_adapter(self):
        from graph_bench.adapters.arangodb import ArangoDBAdapter

        adapter = ArangoDBAdapter()
        assert adapter.name == "ArangoDB"
        assert adapter.connected is False


class TestGrafeoAdapter:
    def test_create_adapter(self):
        from graph_bench.adapters.grafeo import GrafeoAdapter

        adapter = GrafeoAdapter()
        assert adapter.name == "Grafeo"
        assert adapter.connected is False
