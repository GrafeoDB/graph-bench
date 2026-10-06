r"""
Correctness of every Grafeo adapter operation the benchmarks use.

Each test loads a tiny deterministic graph and compares the adapter's answer
with a NetworkX reference, so a benchmark never times a query that fails or
returns a wrong answer. The tests run against embedded Grafeo, and against
grafeo-server (GWP) when one answers (see ``grafeo_server_unavailable`` in
conftest.py for the reason a run skips it).
"""

from __future__ import annotations

from itertools import pairwise

import networkx as nx
import pytest

from graph_bench.adapters.base import BaseAdapter, ldbc_directed_lcc
from graph_bench.benchmarks.query import FILTER_EQUALITY_QUERY, FILTER_RANGE_QUERY
from graph_bench.benchmarks.write import UPDATE_QUERIES
from graph_bench.types import UnsupportedOperation

from .grafeo_dataset import ALL_IDS, CITIES, PEOPLE, load, reference

LDBC_INFINITY = 9223372036854775807


def pagerank_reference(graph: nx.DiGraph, damping: float, iterations: int) -> dict[str, float]:
    """Power iteration with the rank of sink nodes spread evenly (LDBC Graphalytics)."""
    n = graph.number_of_nodes()
    rank = dict.fromkeys(graph, 1.0 / n)
    for _ in range(iterations):
        sinks = sum(rank[v] for v in graph if graph.out_degree(v) == 0)
        rank = {
            v: (1 - damping) / n
            + damping * (sinks / n + sum(rank[u] / graph.out_degree(u) for u in graph.predecessors(v)))
            for v in graph
        }
    return rank


def within(graph: nx.DiGraph, start: str, depth: int) -> set[str]:
    return set(nx.single_source_shortest_path_length(graph, start, cutoff=depth))


@pytest.fixture
def loaded(grafeo_adapter: BaseAdapter) -> BaseAdapter:
    load(grafeo_adapter)
    return grafeo_adapter


def query_errors() -> tuple[type[Exception], ...]:
    """The exceptions a failing query raises on each Grafeo transport."""
    from grafeo import GrafeoError

    errors: tuple[type[Exception], ...] = (GrafeoError,)
    try:
        from graph_bench.adapters.grafeo_server import GrafeoServerError
    except ImportError:
        return errors
    return (*errors, GrafeoServerError)


# ── Storage ─────────────────────────────────────────────────────


def test_load_counts(grafeo_adapter: BaseAdapter) -> None:
    assert load(grafeo_adapter) == (11, 12)
    assert grafeo_adapter.count_nodes() == 11
    assert grafeo_adapter.count_nodes(label="Person") == 9
    assert grafeo_adapter.count_nodes(label="City") == 2
    assert grafeo_adapter.count_edges() == 12
    assert grafeo_adapter.count_edges(edge_type="FOLLOWS") == 9
    assert grafeo_adapter.count_edges(edge_type="LIVES_IN") == 3


def test_insert_edges_counts_only_created_edges(loaded: BaseAdapter) -> None:
    created = loaded.insert_edges([("p0", "nobody", "FOLLOWS", {}), ("p8", "p0", "FOLLOWS", {"weight": 2.5})])
    assert created == 1
    assert loaded.count_edges() == 13
    assert loaded.get_neighbors("p8") == ["p0"]


def test_get_node_returns_exact_properties(loaded: BaseAdapter) -> None:
    for node in PEOPLE + CITIES:
        assert loaded.get_node(node["id"]) == node
    assert loaded.get_node("nobody") is None


def test_update_node(loaded: BaseAdapter) -> None:
    assert loaded.update_node("p1", {"age": 26, "score": 7}) is True
    assert loaded.get_node("p1") == {**PEOPLE[1], "age": 26, "score": 7}
    assert loaded.update_node("nobody", {"score": 1}) is False
    assert loaded.count_nodes() == 11


def test_get_nodes_by_label(loaded: BaseAdapter) -> None:
    people = loaded.get_nodes_by_label("Person", limit=100)
    assert sorted(people, key=lambda n: n["id"]) == PEOPLE
    assert len(loaded.get_nodes_by_label("Person", limit=3)) == 3
    assert sorted(loaded.get_nodes_by_label("City"), key=lambda n: n["id"]) == CITIES


def test_clear_empties_the_database(loaded: BaseAdapter) -> None:
    loaded.clear()
    assert loaded.count_nodes() == 0
    assert loaded.count_edges() == 0
    assert load(loaded) == (11, 12)


# ── Benchmark statements run through execute_query ──────────────


def test_property_update_statement(loaded: BaseAdapter) -> None:
    # The first dialect PropertyUpdateBenchmark tries is the one Grafeo must run.
    loaded.execute_query(UPDATE_QUERIES[0], params={"id": "p4", "score": 42})
    assert loaded.get_node("p4") == {**PEOPLE[4], "score": 42}


@pytest.mark.parametrize(("city", "expected"), [("Utrecht", {"p0", "p1"}), ("Gouda", {"p6", "p7", "p8"}), ("X", set())])
def test_filter_equality_statement(loaded: BaseAdapter, city: str, expected: set[str]) -> None:
    rows = loaded.execute_query(FILTER_EQUALITY_QUERY, params={"city": city})
    assert {row["n"]["id"] for row in rows} == expected
    assert len(rows) == len(expected)


def test_filter_range_statement(loaded: BaseAdapter) -> None:
    rows = loaded.execute_query(FILTER_RANGE_QUERY, params={"min_age": 28, "max_age": 33})
    assert sorted(row["n"]["id"] for row in rows) == ["p0", "p4", "p6", "p7"]


@pytest.mark.parametrize(
    "statement",
    [
        "MATCH (n) RETURN n; MATCH (m) RETURN m",  # several statements in one string
        "MATCH (n:Person {id: $missing}) RETURN n",  # parameter used but not supplied
        "MATCH (n:Person) WITH n.age + 1 RETURN n",  # unnamed expression in WITH
        "MATCH (n {id: $id}) SET n.score = $missing",  # failing write
    ],
)
def test_failing_statement_raises(loaded: BaseAdapter, statement: str) -> None:
    with pytest.raises(query_errors()):
        loaded.execute_query(statement, params={"id": "p0"})
    # The failed statement leaves the adapter usable.
    assert loaded.update_node("p0", {"score": 1}) is True


# ── Traversal ───────────────────────────────────────────────────


def test_get_neighbors(loaded: BaseAdapter) -> None:
    for edge_type in (None, "FOLLOWS", "LIVES_IN"):
        graph = reference(edge_type)
        for node in ALL_IDS:
            assert sorted(loaded.get_neighbors(node, edge_type=edge_type)) == sorted(graph.successors(node))


@pytest.mark.parametrize("edge_type", [None, "FOLLOWS"])
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize("start", ["p0", "p3", "p6", "p8"])
def test_traverse_bfs(loaded: BaseAdapter, start: str, depth: int, edge_type: str | None) -> None:
    visited = loaded.traverse_bfs(start, max_depth=depth, edge_type=edge_type)
    assert visited[0] == start
    assert len(visited) == len(set(visited))
    assert set(visited) == within(reference(edge_type), start, depth)


@pytest.mark.parametrize("edge_type", [None, "FOLLOWS"])
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("start", ["p0", "p3", "p6"])
def test_traverse_dfs(loaded: BaseAdapter, start: str, depth: int, edge_type: str | None) -> None:
    graph = reference(edge_type)
    visited = loaded.traverse_dfs(start, max_depth=depth, edge_type=edge_type)
    assert visited[0] == start
    assert len(visited) == len(set(visited))
    # A depth-limited DFS sees every successor and nothing beyond the depth.
    assert set(graph.successors(start)) <= set(visited) <= within(graph, start, depth)


def test_traverse_dfs_on_a_chain(loaded: BaseAdapter) -> None:
    assert loaded.traverse_dfs("p6", max_depth=3) == ["p6", "p7"]
    assert loaded.traverse_dfs("p6", max_depth=0) == ["p6"]


@pytest.mark.parametrize(
    ("source", "target", "edge_type"),
    [
        ("p0", "p5", None),
        ("p1", "p3", None),
        ("p0", "c1", None),
        ("p1", "p0", "FOLLOWS"),
        ("p0", "c1", "FOLLOWS"),  # needs a LIVES_IN edge
        ("p5", "p0", None),  # unreachable
        ("p8", "p0", None),  # isolated
    ],
)
def test_shortest_path(loaded: BaseAdapter, source: str, target: str, edge_type: str | None) -> None:
    graph = reference(edge_type)
    path = loaded.shortest_path(source, target, edge_type=edge_type)
    if not nx.has_path(graph, source, target):
        assert path is None
        return
    assert path is not None
    assert path[0] == source
    assert path[-1] == target
    assert all(graph.has_edge(u, v) for u, v in pairwise(path))
    assert len(path) - 1 == nx.shortest_path_length(graph, source, target)


# ── Algorithms ──────────────────────────────────────────────────


def test_pagerank_converged(loaded: BaseAdapter) -> None:
    scores = loaded.pagerank(damping=0.85, max_iterations=100, tolerance=1e-12)
    expected = nx.pagerank(reference(), alpha=0.85, weight=None, tol=1e-13, max_iter=1000)
    assert set(scores) == set(ALL_IDS)
    for node, score in expected.items():
        assert scores[node] == pytest.approx(score, abs=1e-6)


def test_pagerank_honours_the_iteration_count(loaded: BaseAdapter) -> None:
    # LdbcPageRankBenchmark asks for exactly 20 iterations; 2 makes a difference visible.
    for iterations in (2, 20):
        scores = loaded.pagerank(damping=0.85, max_iterations=iterations, tolerance=1e-12)
        expected = pagerank_reference(reference(), 0.85, iterations)
        assert scores == pytest.approx(expected, abs=1e-9)


def assert_partition(communities: list[set[str]]) -> None:
    """Every node in exactly one community, and no community spans two components."""
    members = [node for community in communities for node in community]
    assert sorted(members) == sorted(ALL_IDS)
    component_of = {
        node: i for i, component in enumerate(nx.weakly_connected_components(reference())) for node in component
    }
    for community in communities:
        assert len({component_of[node] for node in community}) == 1, community


@pytest.mark.parametrize("algorithm", ["louvain", "label_propagation"])
def test_community_detection(loaded: BaseAdapter, algorithm: str) -> None:
    communities = loaded.community_detection(algorithm=algorithm)
    assert_partition(communities)
    assert {"p8"} in communities
    assert any({"p6", "p7"} <= community for community in communities)
    if algorithm == "louvain":
        assert any({"p3", "p4", "p5"} <= community for community in communities)


def test_community_detection_rejects_unknown_algorithm(loaded: BaseAdapter) -> None:
    with pytest.raises(ValueError, match="Unknown community detection algorithm"):
        loaded.community_detection(algorithm="no_such_algorithm")


def test_weakly_connected_components(loaded: BaseAdapter) -> None:
    components = loaded.weakly_connected_components()
    expected = list(nx.weakly_connected_components(reference()))
    assert sorted(map(sorted, components)) == sorted(map(sorted, expected))


@pytest.mark.parametrize("source", ["p0", "p3", "p8"])
def test_bfs_levels(loaded: BaseAdapter, source: str) -> None:
    levels = loaded.bfs_levels(source)
    expected = nx.single_source_shortest_path_length(reference(), source)
    for node in ALL_IDS:
        if node in expected:
            assert levels[node] == expected[node], node
        else:  # unreachable: left out, or the LDBC infinity sentinel
            assert levels.get(node, LDBC_INFINITY) == LDBC_INFINITY, node


@pytest.mark.parametrize("source", ["p0", "p3"])
def test_sssp(loaded: BaseAdapter, source: str) -> None:
    distances = loaded.sssp(source)
    expected = nx.single_source_dijkstra_path_length(reference(), source, weight="weight")
    for node in ALL_IDS:
        if node in expected:
            assert distances[node] == pytest.approx(expected[node]), node
        else:  # unreachable: left out, or infinity
            assert distances.get(node, float("inf")) == float("inf"), node


def test_local_clustering_coefficient_undirected(loaded: BaseAdapter) -> None:
    # Grafeo computes the undirected coefficient natively, as Neo4j GDS does.
    loaded.take_fallbacks()
    coefficients = loaded.local_clustering_coefficient(directed=False)
    assert loaded.take_fallbacks() == set()
    expected = nx.clustering(reference().to_undirected())
    assert set(coefficients) == set(ALL_IDS)
    for node, value in expected.items():
        assert coefficients[node] == pytest.approx(value), node


def test_local_clustering_coefficient_directed(loaded: BaseAdapter) -> None:
    # The LDBC directed definition has no native Grafeo implementation: it falls back.
    loaded.take_fallbacks()
    coefficients = loaded.local_clustering_coefficient(directed=True)
    assert loaded.take_fallbacks() == {"local_clustering_coefficient"}
    expected = ldbc_directed_lcc(reference())
    assert set(coefficients) == set(ALL_IDS)
    for node, value in expected.items():
        assert coefficients[node] == pytest.approx(value), node
    # p0 has neighbours p1, p2, p3, c0 and edges p1->p2, p2->p3, p1->c0 between them: 3 / (4 * 3)
    assert coefficients["p0"] == pytest.approx(3 / 12)


VECTORS = {
    "v0": [1.0, 0.0, 0.0],
    "v1": [0.9, 0.1, 0.0],
    "v2": [0.0, 1.0, 0.0],
    "v3": [0.0, 0.0, 1.0],
    "v4": [0.7, 0.7, 0.0],
}


def load_vectors(adapter: BaseAdapter) -> None:
    nodes = [{"id": key, "embedding": vec, "cluster": i} for i, (key, vec) in enumerate(VECTORS.items())]
    assert adapter.insert_nodes(nodes, label="VectorNode") == 5


def test_vector_scan(grafeo_adapter: BaseAdapter) -> None:
    load_vectors(grafeo_adapter)
    hits = grafeo_adapter.vector_scan([1.0, 0.05, 0.0], k=3)
    assert [node_id for node_id, _ in hits] == ["v0", "v1", "v4"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-2)
    assert grafeo_adapter.vector_search([1.0, 0.05, 0.0], k=3) == hits


def native_ann(adapter: BaseAdapter) -> None:
    """Build the native index; a server without vector support skips (the reason is checked here)."""
    from graph_bench.adapters.grafeo import GrafeoAdapter

    try:
        adapter.create_vector_index("VectorNode", "embedding", dimensions=3, m=16, ef_construction=64)
    except UnsupportedOperation as unsupported:
        if isinstance(adapter, GrafeoAdapter):
            raise  # embedded Grafeo always has HNSW
        pytest.skip(str(unsupported))


def test_vector_ann_search(grafeo_adapter: BaseAdapter) -> None:
    # Loaded through insert_nodes (direct API or vector() on a parameter): the index must hold them all.
    load_vectors(grafeo_adapter)
    native_ann(grafeo_adapter)
    grafeo_adapter.take_fallbacks()
    hits = grafeo_adapter.vector_ann_search([1.0, 0.05, 0.0], k=3, ef=16)
    assert grafeo_adapter.take_fallbacks() == set()
    assert [node_id for node_id, _ in hits] == ["v0", "v1", "v4"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-2)
    assert len(grafeo_adapter.vector_ann_search([0.0, 0.0, 1.0], k=5, ef=16)) == 5


def test_deleted_vectors_leave_the_ann_results(grafeo_adapter: BaseAdapter) -> None:
    load_vectors(grafeo_adapter)
    native_ann(grafeo_adapter)
    assert grafeo_adapter.delete_nodes(["v0", "nobody"]) == 1
    assert grafeo_adapter.get_node("v0") is None
    ids = [node_id for node_id, _ in grafeo_adapter.vector_ann_search([1.0, 0.05, 0.0], k=3, ef=16)]
    assert "v0" not in ids
    assert ids[0] == "v1"


def test_delete_nodes_removes_their_edges(loaded: BaseAdapter) -> None:
    assert loaded.delete_nodes(["p3"]) == 1
    assert loaded.count_nodes() == 10
    assert loaded.count_edges() == 12 - 4  # p2->p3, p0->p3, p3->p4, p5->p3
