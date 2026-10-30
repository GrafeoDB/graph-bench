r"""
A tiny deterministic graph for the Grafeo correctness tests, and its NetworkX reference.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from graph_bench.adapters.base import BaseAdapter

PEOPLE: list[dict[str, Any]] = [
    {"id": "p0", "name": "Alix", "age": 30, "city": "Utrecht"},
    {"id": "p1", "name": "Gus", "age": 25, "city": "Utrecht"},
    {"id": "p2", "name": "Vincent", "age": 41, "city": "Leiden"},
    {"id": "p3", "name": "Jules", "age": 35, "city": "Leiden"},
    {"id": "p4", "name": "Mia", "age": 28, "city": "Delft"},
    {"id": "p5", "name": "Butch", "age": 52, "city": "Delft"},
    {"id": "p6", "name": "Django", "age": 33, "city": "Gouda"},
    {"id": "p7", "name": "Shosanna", "age": 29, "city": "Gouda"},
    {"id": "p8", "name": "Hans", "age": 60, "city": "Gouda"},  # isolated
]
CITIES: list[dict[str, Any]] = [{"id": "c0", "name": "Utrecht"}, {"id": "c1", "name": "Leiden"}]
# Two triangles joined by p2->p3 and p0->p3, a separate pair and an isolated node.
# The direct p0->p3 edge is heavier than the path through p1 and p2.
FOLLOWS = [
    ("p0", "p1", 1.0),
    ("p1", "p2", 2.0),
    ("p2", "p0", 1.0),
    ("p2", "p3", 4.0),
    ("p0", "p3", 9.0),
    ("p3", "p4", 1.0),
    ("p4", "p5", 1.0),
    ("p5", "p3", 1.0),
    ("p6", "p7", 1.0),
]
LIVES_IN = [("p0", "c0"), ("p1", "c0"), ("p2", "c1")]
ALL_IDS = [n["id"] for n in PEOPLE + CITIES]


def load(adapter: BaseAdapter) -> tuple[int, int]:
    """Load the test graph and return (nodes inserted, edges inserted)."""
    nodes = adapter.insert_nodes(PEOPLE, label="Person") + adapter.insert_nodes(CITIES, label="City")
    edges = [(s, t, "FOLLOWS", {"weight": w}) for s, t, w in FOLLOWS]
    edges += [(s, t, "LIVES_IN", {}) for s, t in LIVES_IN]
    return nodes, adapter.insert_edges(edges)


def reference(edge_type: str | None = None) -> nx.DiGraph:
    """The test graph in NetworkX (edges without a weight weigh 1, as in Grafeo)."""
    graph = nx.DiGraph()
    graph.add_nodes_from(ALL_IDS)
    if edge_type in (None, "FOLLOWS"):
        graph.add_weighted_edges_from(FOLLOWS)
    if edge_type in (None, "LIVES_IN"):
        graph.add_edges_from(LIVES_IN)
    return graph
