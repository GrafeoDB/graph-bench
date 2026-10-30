"""Graphalytics BFS, WCC and PageRank on the symmetric KNOWS graph, with NetworkX references."""

from collections.abc import Mapping
from typing import Any

import networkx as nx

from graph_bench.answers import LDBC_EPS, SNB_EPS, canonical_map
from graph_bench.datasets.schema import Dataset
from graph_bench.suites import Algorithm

ALGORITHMS: tuple[Algorithm, ...] = (
    Algorithm("graphalytics_bfs", "BFS", "algorithms", "property_graph", "graphalytics", "bfs", SNB_EPS),
    Algorithm("graphalytics_wcc", "WCC", "algorithms", "property_graph", "graphalytics", "wcc", SNB_EPS),
    Algorithm("graphalytics_pr", "PageRank", "algorithms", "property_graph", "graphalytics", "pr", LDBC_EPS),
)

_BY_ID = {a.id: a for a in ALGORITHMS}


def _graph(ds: Dataset) -> nx.DiGraph:
    """The dataset as a directed graph; E is stored in both directions, so this is the undirected graph."""
    vertices = next(t for t in ds.nodes if t.label == "V")
    edges = next(t for t in ds.edges if t.type == "E")
    g = nx.DiGraph()
    g.add_nodes_from(row["id"] for row in ds.rows(vertices))
    g.add_edges_from((row["src"], row["dst"]) for row in ds.rows(edges))
    return g


def min_member(components: Mapping[Any, Any]) -> dict[int, int]:
    """Relabel a vertex -> component label map so that each component is labelled by its smallest vertex.

    Databases label components arbitrarily; this makes every labelling of the same partition identical.
    """
    smallest: dict[Any, int] = {}
    for vertex, label in components.items():
        v = int(vertex)
        if label not in smallest or v < smallest[label]:
            smallest[label] = v
    return {int(vertex): smallest[label] for vertex, label in components.items()}


def bindings(algo_id: str, ds: Dataset) -> list[dict[str, Any]]:
    """BFS starts at the vertex with the highest degree (lowest id on ties); WCC and PR take no parameters."""
    if _BY_ID[algo_id].name != "bfs":
        return [{}]
    g = _graph(ds)
    source = min(g.nodes, key=lambda v: (-g.out_degree(v), v))
    return [{"source": source}]


def reference(algo_id: str, ds: Dataset, binding: dict[str, Any]) -> dict[int, Any]:
    """The canonical NetworkX reference: BFS depths (unreachable dropped), WCC min member, PageRank."""
    name = _BY_ID[algo_id].name
    g = _graph(ds)
    if name == "bfs":
        depths = nx.single_source_shortest_path_length(g, binding["source"])
        return canonical_map(depths, drop_unreachable=True)
    if name == "wcc":
        labels = {v: i for i, comp in enumerate(nx.weakly_connected_components(g)) for v in comp}
        return canonical_map(min_member(labels), drop_unreachable=False)
    ranks = nx.pagerank(g, alpha=0.85, tol=1e-10, max_iter=1000)
    return canonical_map(ranks, drop_unreachable=False)
