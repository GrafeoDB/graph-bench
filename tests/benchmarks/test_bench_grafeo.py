r"""
CodSpeed benchmarks for the embedded Grafeo adapter.

These exercise the GrafeoAdapter code paths used by the graph-bench suite
(bulk loading, point lookups, traversals and native graph algorithms) on an
in-memory database. They are skipped when grafeo (or grpcio, imported by the
adapters package) is not installed.

Run with:
    uv sync --extra dev --extra grafeo
    uv run --with grpcio --with protobuf pytest tests/benchmarks --codspeed --no-cov
"""

import pytest

pytest.importorskip("grafeo")
pytest.importorskip("grpc")

from graph_bench.adapters.grafeo import GrafeoAdapter
from graph_bench.datasets.synthetic import SyntheticSocialNetwork
from graph_bench.types import ScaleConfig

SCALE = ScaleConfig(name="bench", nodes=1_000, edges=5_000)
START_NODES = [f"person_{i}" for i in range(0, 200, 10)]


@pytest.fixture(scope="module")
def dataset():
    return SyntheticSocialNetwork(seed=42, use_faker=False).generate(SCALE)


def _new_adapter() -> GrafeoAdapter:
    adapter = GrafeoAdapter()
    adapter.connect(uri=":memory:")
    return adapter


@pytest.fixture(scope="module")
def loaded(dataset):
    nodes, edges = dataset
    adapter = _new_adapter()
    adapter.insert_nodes(nodes, label="Person")
    adapter.insert_edges(edges)
    yield adapter
    adapter.disconnect()


def test_insert_nodes(benchmark, dataset):
    nodes, _ = dataset

    def setup():
        return (_new_adapter(), nodes), {"label": "Person"}

    def run(adapter, nodes, label):
        return adapter.insert_nodes(nodes, label=label)

    count = benchmark.pedantic(run, setup=setup, rounds=1)
    assert count == len(nodes)


def test_insert_edges(benchmark, dataset):
    nodes, edges = dataset

    def setup():
        adapter = _new_adapter()
        adapter.insert_nodes(nodes, label="Person")
        return (adapter, edges), {}

    def run(adapter, edges):
        return adapter.insert_edges(edges)

    count = benchmark.pedantic(run, setup=setup, rounds=1)
    assert count == len(edges)


@pytest.mark.benchmark
def test_get_node(loaded):
    for node_id in START_NODES:
        loaded.get_node(node_id)


@pytest.mark.benchmark
def test_get_nodes_by_label(loaded):
    assert len(loaded.get_nodes_by_label("Person", limit=100)) == 100


@pytest.mark.benchmark
def test_hop_1_neighbors(loaded):
    total = sum(len(loaded.get_neighbors(node_id)) for node_id in START_NODES)
    assert total > 0


@pytest.mark.benchmark
def test_hop_2_bfs(loaded):
    total = sum(len(loaded.traverse_bfs(node_id, max_depth=2)) for node_id in START_NODES[:5])
    assert total > 0


@pytest.mark.benchmark
def test_dfs_depth_3(loaded):
    total = sum(len(loaded.traverse_dfs(node_id, max_depth=3)) for node_id in START_NODES[:5])
    assert total > 0


@pytest.mark.benchmark
def test_two_hop_pattern_query(loaded):
    rows = loaded.execute_query(
        "MATCH (a:Person {id: $id})-[:FOLLOWS]->(b:Person)-[:FOLLOWS]->(c:Person) RETURN count(c) AS c",
        params={"id": "person_5"},
    )
    assert rows


@pytest.mark.benchmark
def test_count_nodes_and_edges(loaded):
    assert loaded.count_nodes(label="Person") == SCALE.nodes
    assert loaded.count_edges() == SCALE.edges


@pytest.mark.benchmark
def test_pagerank(loaded):
    assert loaded.pagerank()


@pytest.mark.benchmark
def test_weakly_connected_components(loaded):
    assert loaded.weakly_connected_components()


@pytest.mark.benchmark
def test_louvain(loaded):
    assert loaded.community_detection(algorithm="louvain")


@pytest.mark.benchmark
def test_local_clustering_coefficient(loaded):
    assert loaded.local_clustering_coefficient()


@pytest.mark.benchmark
def test_bfs_levels(loaded):
    assert loaded.bfs_levels("person_5")
