r"""
An in-memory adapter without native algorithms, to run the BaseAdapter fallbacks.
"""

from __future__ import annotations

from graph_bench.adapters.base import BaseAdapter


class DictAdapter(BaseAdapter):
    """An in-memory adapter without native algorithms, to run the BaseAdapter fallbacks."""

    def __init__(self, edges: list[tuple[str, str]] | None = None) -> None:
        self._labels: dict[str, str] = {}
        self._edges: list[tuple[str, str]] = []
        for src, tgt in edges or []:
            self._labels.setdefault(src, "Person")
            self._labels.setdefault(tgt, "Person")
            self._edges.append((src, tgt))

    @property
    def name(self) -> str:
        return "dict"

    def connect(self, *, uri=None, **kwargs):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def clear(self):
        self._labels = {}
        self._edges = []

    def insert_nodes(self, nodes, *, label="Node", batch_size=1000):
        for node in nodes:
            self._labels[node["id"]] = label
        return len(nodes)

    def get_node(self, node_id):
        return {"id": node_id} if node_id in self._labels else None

    def update_node(self, node_id, properties):
        return node_id in self._labels

    def get_nodes_by_label(self, label, *, limit=100):
        return [{"id": n} for n, node_label in self._labels.items() if node_label == label][:limit]

    def insert_edges(self, edges, *, batch_size=1000):
        created = [(s, t) for s, t, _, _ in edges if s in self._labels and t in self._labels]
        self._edges.extend(created)
        return len(created)

    def get_neighbors(self, node_id, *, edge_type=None):
        return [t for s, t in self._edges if s == node_id]

    def shortest_path(self, source, target, *, edge_type=None, weighted=False):
        return None

    def execute_query(self, query, *, params=None):
        return []

    def count_nodes(self, *, label=None):
        return sum(1 for node_label in self._labels.values() if label in (None, node_label))

    def count_edges(self, *, edge_type=None):
        return len(self._edges)
