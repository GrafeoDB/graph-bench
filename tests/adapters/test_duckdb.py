r"""Tests for DuckDB adapter."""

import pytest

from graph_bench.adapters.duckdb import DuckDBAdapter


class TestDuckDBAdapter:
    """Tests for DuckDB embedded database adapter."""

    @pytest.fixture
    def adapter(self):
        """Create in-memory DuckDB adapter."""
        adapter = DuckDBAdapter()
        adapter.connect(uri=":memory:")
        yield adapter
        adapter.disconnect()

    def test_name(self, adapter):
        assert adapter.name == "DuckDB"

    def test_version(self, adapter):
        # Should return a version string
        assert adapter.version != "unknown"

    def test_connected(self, adapter):
        assert adapter.connected is True

    def test_insert_and_get_node(self, adapter):
        nodes = [{"id": "n1", "name": "Alice", "age": 30}]
        count = adapter.insert_nodes(nodes, label="Person")

        assert count == 1

        node = adapter.get_node("n1")
        assert node is not None
        assert node["id"] == "n1"
        assert node["name"] == "Alice"
        assert node["age"] == 30

    def test_insert_nodes_batch(self, adapter):
        nodes = [{"id": f"n{i}", "value": i} for i in range(100)]
        count = adapter.insert_nodes(nodes, label="Node", batch_size=25)

        assert count == 100
        assert adapter.count_nodes() == 100

    def test_get_nodes_by_label(self, adapter):
        nodes = [
            {"id": "p1", "name": "Alice"},
            {"id": "p2", "name": "Bob"},
            {"id": "c1", "name": "NYC"},
        ]
        adapter.insert_nodes(nodes[:2], label="Person")
        adapter.insert_nodes(nodes[2:], label="City")

        persons = adapter.get_nodes_by_label("Person")
        assert len(persons) == 2

        cities = adapter.get_nodes_by_label("City")
        assert len(cities) == 1

    def test_insert_and_get_edges(self, adapter):
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        adapter.insert_nodes(nodes, label="Node")

        edges = [
            ("a", "b", "KNOWS", {"since": 2020}),
            ("b", "c", "KNOWS", {}),
        ]
        count = adapter.insert_edges(edges)

        assert count == 2
        assert adapter.count_edges() == 2

    def test_get_neighbors(self, adapter):
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        adapter.insert_nodes(nodes, label="Node")

        edges = [
            ("a", "b", "FOLLOWS", {}),
            ("a", "c", "KNOWS", {}),
        ]
        adapter.insert_edges(edges)

        neighbors = adapter.get_neighbors("a")
        assert set(neighbors) == {"b", "c"}

        follows_neighbors = adapter.get_neighbors("a", edge_type="FOLLOWS")
        assert follows_neighbors == ["b"]

    def test_shortest_path(self, adapter):
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
        adapter.insert_nodes(nodes, label="Node")

        edges = [
            ("a", "b", "EDGE", {}),
            ("b", "c", "EDGE", {}),
            ("c", "d", "EDGE", {}),
        ]
        adapter.insert_edges(edges)

        path = adapter.shortest_path("a", "d")
        assert path == ["a", "b", "c", "d"]

    def test_shortest_path_no_path(self, adapter):
        nodes = [{"id": "a"}, {"id": "b"}]
        adapter.insert_nodes(nodes, label="Node")
        # No edges

        path = adapter.shortest_path("a", "b")
        assert path is None

    def test_traverse_bfs(self, adapter):
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        adapter.insert_nodes(nodes, label="Node")

        edges = [("a", "b", "E", {}), ("b", "c", "E", {})]
        adapter.insert_edges(edges)

        visited = adapter.traverse_bfs("a", max_depth=2)
        assert set(visited) == {"a", "b", "c"}

    def test_traverse_dfs(self, adapter):
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        adapter.insert_nodes(nodes, label="Node")

        edges = [("a", "b", "E", {}), ("b", "c", "E", {})]
        adapter.insert_edges(edges)

        visited = adapter.traverse_dfs("a", max_depth=2)
        assert set(visited) == {"a", "b", "c"}

    def test_execute_query(self, adapter):
        nodes = [{"id": "n1", "value": 10}, {"id": "n2", "value": 20}]
        adapter.insert_nodes(nodes, label="Node")

        results = adapter.execute_query("SELECT COUNT(*) as cnt FROM nodes")
        assert results[0]["cnt"] == 2

    def test_count_nodes_by_label(self, adapter):
        adapter.insert_nodes([{"id": "p1"}], label="Person")
        adapter.insert_nodes([{"id": "p2"}], label="Person")
        adapter.insert_nodes([{"id": "c1"}], label="City")

        assert adapter.count_nodes() == 3
        assert adapter.count_nodes(label="Person") == 2
        assert adapter.count_nodes(label="City") == 1

    def test_count_edges_by_type(self, adapter):
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        adapter.insert_nodes(nodes, label="Node")

        edges = [
            ("a", "b", "KNOWS", {}),
            ("a", "c", "FOLLOWS", {}),
        ]
        adapter.insert_edges(edges)

        assert adapter.count_edges() == 2
        assert adapter.count_edges(edge_type="KNOWS") == 1
        assert adapter.count_edges(edge_type="FOLLOWS") == 1

    def test_clear(self, adapter):
        nodes = [{"id": "n1"}]
        adapter.insert_nodes(nodes, label="Node")
        edges = [("n1", "n1", "SELF", {})]
        adapter.insert_edges(edges)

        assert adapter.count_nodes() > 0
        assert adapter.count_edges() > 0

        adapter.clear()

        assert adapter.count_nodes() == 0
        assert adapter.count_edges() == 0

    def test_context_manager(self):
        with DuckDBAdapter() as adapter:
            adapter.connect(uri=":memory:")
            assert adapter.connected is True
        # After context exit, should be disconnected
        assert adapter.connected is False

    def test_pagerank_networkx_fallback(self, adapter):
        """PageRank uses NetworkX fallback for databases without native support."""
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        adapter.insert_nodes(nodes, label="Person")
        edges = [("a", "b", "E", {}), ("b", "c", "E", {}), ("c", "a", "E", {})]
        adapter.insert_edges(edges)

        scores = adapter.pagerank()
        assert len(scores) == 3
        assert all(0 <= v <= 1 for v in scores.values())

    def test_community_detection_networkx_fallback(self, adapter):
        """Community detection uses NetworkX fallback for databases without native support."""
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        adapter.insert_nodes(nodes, label="Person")
        edges = [("a", "b", "E", {}), ("b", "c", "E", {})]
        adapter.insert_edges(edges)

        communities = adapter.community_detection()
        assert isinstance(communities, list)
        # All nodes should be in some community
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)
        assert all_nodes == {"a", "b", "c"}
