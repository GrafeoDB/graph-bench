r"""
The two local clustering coefficient definitions and their reference.

``ldbc_lcc`` uses the LDBC Graphalytics definition for directed graphs:
N(v) holds the in- and out-neighbours, the directed edges between them are
counted, and the count is divided by d * (d - 1). ``lcc_undirected`` takes every
edge as undirected. ``nx.clustering`` on a DiGraph (Fagiolo) is neither.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graph_bench.adapters.base import ldbc_directed_lcc

from .dict_adapter import DictAdapter

TRIANGLE = [("0", "1"), ("1", "2"), ("2", "0")]


def test_directed_cycle_counts_each_direction() -> None:
    # 0->1->2->0: node 0 has neighbours {1, 2} and one directed edge between them (1->2).
    assert ldbc_directed_lcc(nx.DiGraph(TRIANGLE)) == {"0": 0.5, "1": 0.5, "2": 0.5}
    assert nx.clustering(nx.Graph(TRIANGLE)) == {"0": 1.0, "1": 1.0, "2": 1.0}


def test_reciprocal_edges_count_twice() -> None:
    graph = nx.DiGraph([*TRIANGLE, ("2", "1")])
    # Node 0: neighbours {1, 2}, edges 1->2 and 2->1: 2 / (2 * 1).
    assert ldbc_directed_lcc(graph)["0"] == 1.0
    # Node 1: neighbours {0, 2}, edge 2->0 only: 1 / 2.
    assert ldbc_directed_lcc(graph)["1"] == 0.5


def test_degree_below_two_is_zero() -> None:
    graph = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "b")])
    graph.add_node("c")
    # a and c have fewer than two neighbours; the self-loop does not count.
    assert ldbc_directed_lcc(graph) == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_star_with_one_link_between_leaves() -> None:
    # Hub h -> a, b, c and a -> b: h has d = 3 and 1 edge: 1 / 6.
    graph = nx.DiGraph([("h", "a"), ("h", "b"), ("h", "c"), ("a", "b")])
    assert ldbc_directed_lcc(graph)["h"] == pytest.approx(1 / 6)
    assert nx.clustering(nx.Graph(graph))["h"] == pytest.approx(1 / 3)


def test_differs_from_the_fagiolo_coefficient() -> None:
    # a <-> b, a -> c, d -> a, d -> b: N(a) = {b, c, d} and one edge between them (d -> b).
    graph = nx.DiGraph([("a", "c"), ("b", "a"), ("a", "b"), ("d", "a"), ("d", "b")])
    assert ldbc_directed_lcc(graph)["a"] == pytest.approx(1 / 6)
    assert nx.clustering(graph)["a"] == pytest.approx(0.2)  # Fagiolo counts the reciprocal pair


@pytest.mark.parametrize(("directed", "expected"), [(True, 0.5), (False, 1.0)])
def test_base_fallback_follows_the_definition(directed: bool, expected: float) -> None:
    adapter = DictAdapter(TRIANGLE)
    assert adapter.local_clustering_coefficient(directed=directed) == dict.fromkeys(["0", "1", "2"], expected)
    assert adapter.take_fallbacks() == {"local_clustering_coefficient"}
    assert adapter.take_fallbacks() == set()


@pytest.mark.parametrize(("directed", "expected", "fallbacks"), [(True, 0.5, 1), (False, 1.0, 0)])
def test_grafeo_computes_undirected_natively(directed: bool, expected: float, fallbacks: int) -> None:
    from graph_bench.adapters.grafeo import GrafeoAdapter

    adapter = GrafeoAdapter()
    adapter.connect(path=":memory:")
    adapter.insert_nodes([{"id": n} for n in "012"], label="Person")
    adapter.insert_edges([(s, t, "E", {}) for s, t in TRIANGLE])
    adapter.take_fallbacks()
    assert adapter.local_clustering_coefficient(directed=directed) == dict.fromkeys(["0", "1", "2"], expected)
    assert len(adapter.take_fallbacks()) == fallbacks
    adapter.disconnect()
