r"""
Grafeo database adapter using GQL (ISO graph query language).

Grafeo is an embedded graph database with Python bindings.
Uses GQL as the default query language (ISO/IEC 39075).

Requires: uv sync --extra grafeo (grafeo>=0.6.0b1,<0.7)

Environment variables:
    GRAPH_BENCH_GRAFEO_PATH: Database file (default: in-memory). A path without
        the ``.grafeo`` suffix gets it appended, so the database is a single
        file: grafeo 0.6 deprecates the WAL-directory layout and migrates it.

    from graph_bench.adapters.grafeo import GrafeoAdapter

    adapter = GrafeoAdapter()
    adapter.connect(path="./data/bench.grafeo")

    # GQL queries
    adapter.execute_query("MATCH (n:Person) RETURN n.name")

Query errors raise ``grafeo.GrafeoError`` and are never swallowed: a benchmark
whose query fails is recorded as FAILED instead of being timed.
"""

from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["GrafeoAdapter"]

# Every inserted node also gets this label, so lookups by `id` use one label.
_NODE_LABEL = "Node"
_FILE_SUFFIX = ".grafeo"
_INSTALL_HINT = 'grafeo package not installed. Install with: uv pip install "grafeo>=0.6.0b1,<0.7"'

_LOOKUP_NID = "MATCH (n:Node {id: $id}) RETURN id(n) AS nid"
_LOOKUP_NIDS = "UNWIND $keys AS key MATCH (n:Node {id: key}) RETURN key, id(n) AS nid"
_DELETE_NODES = "UNWIND $keys AS key MATCH (n:Node {id: key}) DETACH DELETE n"

# Projection names are database-wide: each traversal takes its own.
_PROJECTION_IDS = itertools.count()


def _database_file(path: str) -> Path:
    """Return the single-file database path for a configured path."""
    db_path = Path(path)
    if db_path.suffix != _FILE_SUFFIX:
        db_path = db_path.with_name(db_path.name + _FILE_SUFFIX)
    return db_path


def _strip_internal(node: Any) -> dict[str, Any]:
    """Strip the internal fields (_id, _labels) from a node map."""
    if isinstance(node, dict):
        return {k: v for k, v in node.items() if k not in ("_id", "_labels")}
    return {}


@AdapterRegistry.register("grafeo")
class GrafeoAdapter(BaseAdapter):
    """Grafeo embedded graph database adapter.

    Loads data with the batch APIs (``batch_create_nodes_with_props`` and
    ``batch_create_edges``; lists of floats are stored as vectors), reads with
    GQL, and runs Grafeo's native algorithms and HNSW vector index. Native
    results use internal node IDs, which are mapped back to the benchmark's
    ``id`` property.

    A BFS or shortest path restricted to one edge type runs natively on a
    projection of that edge type (grafeo 0.6 algorithms take ``projection=``).

    Runs in Python (and is labelled a fallback): depth-limited DFS (the native
    DFS has no depth limit) and the directed (LDBC) clustering coefficient.
    """

    # algorithms.local_clustering_coefficient() ignores edge direction.
    native_lcc_definitions = frozenset({"undirected"})

    def __init__(self) -> None:
        self._db: Any = None
        self._path: Path | None = None
        self._connected = False
        self._vector_metric = "cosine"

    @property
    def name(self) -> str:
        return "Grafeo"

    @property
    def version(self) -> str:
        try:
            import grafeo

            return grafeo.__version__
        except ImportError:
            return "unknown"

    @property
    def is_embedded(self) -> bool:
        return True

    @property
    def path(self) -> Path | None:
        """Database file, or None for an in-memory database."""
        return self._path

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from grafeo import GrafeoDB
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        path = uri or kwargs.get("path") or get_env("GRAFEO_PATH")
        if path and path != ":memory:":
            self._path = _database_file(path)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._db = GrafeoDB(str(self._path))
        else:
            self._path = None
            self._db = GrafeoDB()
        self._connected = True

    def disconnect(self) -> None:
        if self._db is not None:
            self._db.close()
        self._db = None
        self._connected = False

    def clear(self) -> None:
        if self._path is not None:
            # A file database keeps its file and its index; only the data goes.
            self._db.execute("MATCH (n) DETACH DELETE n")
            return
        # In memory: drop the database and start a new one, as the FalkorDB and
        # Grafeo Server adapters drop and recreate their graph.
        from grafeo import GrafeoDB

        if self._db is not None:
            self._db.close()
        self._db = GrafeoDB()

    # ── Writes ──────────────────────────────────────────────────

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        if not nodes:
            return 0
        labels = [label, _NODE_LABEL] if label != _NODE_LABEL else [_NODE_LABEL]
        count = 0
        for start in range(0, len(nodes), batch_size):
            batch = [dict(node) for node in nodes[start : start + batch_size]]
            count += len(self._db.batch_create_nodes_with_props(labels, batch))
        # FAIRNESS.md allows one index on `id`, created after the insert.
        if not self._db.has_property_index("id"):
            self._db.create_property_index("id")
        return count

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        result = self._db.execute(
            "MATCH (n:Node {id: $id}) SET n += $props RETURN n.id AS id",
            {"id": node_id, "props": dict(properties)},
        )
        return any(True for _ in result)

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        """Insert edges between existing nodes and return how many were created.

        Resolves the distinct endpoints through the `id` index with one
        ``UNWIND ... MATCH`` query per batch, then creates the edges with
        ``batch_create_edges``: the same work as a Cypher ``UNWIND`` + ``MATCH``
        + ``CREATE``, split in two calls. Like that query, an edge whose endpoint
        does not exist is not created and not counted.
        """
        if not edges:
            return 0
        endpoints = {key for src, tgt, _, _ in edges for key in (src, tgt)}
        nids = self._internal_ids(endpoints, batch_size=batch_size)
        rows = [
            (nids[src], nids[tgt], edge_type, dict(props))
            for src, tgt, edge_type, props in edges
            if src in nids and tgt in nids
        ]
        count = 0
        for start in range(0, len(rows), batch_size):
            count += len(self._db.batch_create_edges(rows[start : start + batch_size]))
        return count

    # ── Reads ───────────────────────────────────────────────────

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        for row in self._db.execute("MATCH (n:Node {id: $id}) RETURN n", {"id": node_id}):
            return _strip_internal(row["n"])
        return None

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        result = self._db.execute(f"MATCH (n:{label}) RETURN n LIMIT $limit", {"limit": int(limit)})
        return [_strip_internal(row["n"]) for row in result]

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        rel = f"[:{edge_type}]" if edge_type else "[]"
        result = self._db.execute(f"MATCH (n:Node {{id: $id}})-{rel}->(m) RETURN m.id AS id", {"id": node_id})
        return [row["id"] for row in result if row["id"] is not None]

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        result = self._db.execute(query, params or {})
        return [dict(row) for row in result]

    def count_nodes(self, *, label: str | None = None) -> int:
        query = f"MATCH (n:{label}) RETURN count(n) AS count" if label else "MATCH (n) RETURN count(n) AS count"
        for row in self._db.execute(query):
            return row["count"]
        return 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        rel = f"[r:{edge_type}]" if edge_type else "[r]"
        for row in self._db.execute(f"MATCH ()-{rel}->() RETURN count(r) AS count"):
            return row["count"]
        return 0

    # ── ID mapping ──────────────────────────────────────────────

    def _internal_id(self, app_id: str) -> int | None:
        for row in self._db.execute(_LOOKUP_NID, {"id": app_id}):
            return row["nid"]
        return None

    def _internal_ids(self, app_ids: Iterable[str], *, batch_size: int = 1000) -> dict[str, int]:
        """Map benchmark IDs to internal node IDs (missing IDs are left out)."""
        keys = list(app_ids)
        found: dict[str, int] = {}
        for start in range(0, len(keys), batch_size):
            for row in self._db.execute(_LOOKUP_NIDS, {"keys": keys[start : start + batch_size]}):
                found[row["key"]] = row["nid"]
        return found

    def _app_ids(self, nids: Iterable[int]) -> dict[int, str]:
        """Map internal node IDs to benchmark IDs with one batch read."""
        nid_list = list(nids)
        values = self._db.get_property_batch(nid_list, "id")
        return {nid: str(value if value is not None else nid) for nid, value in zip(nid_list, values, strict=True)}

    def _by_app_id(self, values: dict[int, Any]) -> dict[str, Any]:
        app = self._app_ids(values)
        return {app[nid]: value for nid, value in values.items()}

    def _groups(self, labels: dict[int, Any]) -> list[set[str]]:
        """Turn a node-to-label map into sets of benchmark IDs."""
        groups: dict[Any, set[str]] = {}
        for app_id, label in self._by_app_id(labels).items():
            groups.setdefault(label, set()).add(app_id)
        return list(groups.values())

    # ── Traversal ───────────────────────────────────────────────

    @contextmanager
    def _edge_type_view(self, edge_type: str | None) -> Generator[str | None]:
        """Yield a projection with only ``edge_type`` edges (None: the whole graph).

        A projection is a filtered view, not a copy. It is created and dropped
        inside the timed call, so its cost is measured and nothing outlives it.
        """
        if edge_type is None:
            yield None
            return
        name = f"graph_bench_edges_{next(_PROJECTION_IDS)}"
        self._db.create_projection(name, edge_types=[edge_type])
        try:
            yield name
        finally:
            self._db.drop_projection(name)

    def traverse_bfs(
        self,
        start: str,
        *,
        max_depth: int = 3,
        edge_type: str | None = None,
    ) -> list[str]:
        start_nid = self._internal_id(start)
        if start_nid is None or not hasattr(self._db.algorithms, "bfs_layers"):
            return super().traverse_bfs(start, max_depth=max_depth, edge_type=edge_type)
        with self._edge_type_view(edge_type) as projection:
            layers = self._db.algorithms.bfs_layers(start_nid, projection=projection)[: max_depth + 1]
        visited = [nid for layer in layers for nid in layer]
        app = self._app_ids(visited)
        return [app[nid] for nid in visited]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        nids = self._internal_ids([source, target])
        if source not in nids or target not in nids:
            return None
        weight = "weight" if weighted else None
        with self._edge_type_view(edge_type) as projection:
            found = self._db.algorithms.dijkstra(nids[source], nids[target], weight=weight, projection=projection)
        if found is None:
            return None
        _, path = found
        app = self._app_ids(path)
        return [app[nid] for nid in path]

    # ── Algorithms (native, results keyed by benchmark ID) ──────

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        if not hasattr(self._db.algorithms, "pagerank"):
            return super().pagerank(damping=damping, max_iterations=max_iterations, tolerance=tolerance)
        scores = self._db.algorithms.pagerank(damping=damping, max_iterations=max_iterations, tolerance=tolerance)
        return {app_id: float(score) for app_id, score in self._by_app_id(scores).items()}

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        algorithms = self._db.algorithms
        if algorithm == "louvain" and hasattr(algorithms, "louvain"):
            # louvain() returns {"communities": {node: community}, "modularity": ..., ...}
            return self._groups(algorithms.louvain()["communities"])
        if algorithm == "label_propagation" and hasattr(algorithms, "label_propagation"):
            return self._groups(algorithms.label_propagation())
        return super().community_detection(algorithm=algorithm)

    def bfs_levels(self, source: str) -> dict[str, int]:
        """LDBC BFS with native bfs_layers; unreachable vertices are left out."""
        src = self._internal_id(source)
        if src is None or not hasattr(self._db.algorithms, "bfs_layers"):
            return super().bfs_levels(source)
        levels = {nid: depth for depth, layer in enumerate(self._db.algorithms.bfs_layers(src)) for nid in layer}
        return self._by_app_id(levels)

    def weakly_connected_components(self) -> list[set[str]]:
        """LDBC WCC with native connected_components (edges taken as undirected)."""
        if not hasattr(self._db.algorithms, "connected_components"):
            return super().weakly_connected_components()
        return self._groups(self._db.algorithms.connected_components())

    def sssp(self, source: str, *, weight_attr: str = "weight") -> dict[str, float]:
        """LDBC SSSP with native Dijkstra; unreachable vertices are left out.

        ``algorithms.sssp()`` keys its result by the ``name`` property, so the
        adapter uses ``dijkstra()``, which keys it by node.
        """
        src = self._internal_id(source)
        if src is None or not hasattr(self._db.algorithms, "dijkstra"):
            return super().sssp(source, weight_attr=weight_attr)
        distances = self._db.algorithms.dijkstra(src, weight=weight_attr)
        return {app_id: float(distance) for app_id, distance in self._by_app_id(distances).items()}

    def local_clustering_coefficient(self, *, directed: bool = True) -> dict[str, float]:
        """LCC with native local_clustering_coefficient (undirected only).

        Grafeo, like Neo4j GDS, computes the undirected coefficient; the LDBC
        directed definition uses the BaseAdapter fallback.
        """
        definition = "directed" if directed else "undirected"
        if definition not in self.native_lcc_definitions or not hasattr(
            self._db.algorithms, "local_clustering_coefficient"
        ):
            return super().local_clustering_coefficient(directed=directed)
        coefficients = self._db.algorithms.local_clustering_coefficient()
        return {app_id: float(value) for app_id, value in self._by_app_id(coefficients).items()}

    # ── Native vector search (HNSW) ─────────────────────────────

    def create_vector_index(
        self,
        label: str,
        property_name: str,
        *,
        dimensions: int = 128,
        metric: str = "cosine",
        m: int | None = None,
        ef_construction: int | None = None,
        quantization: str | None = None,
    ) -> None:
        """Build Grafeo's HNSW index over the vectors already loaded (and later ones)."""
        if not hasattr(self._db, "create_vector_index"):
            return super().create_vector_index(label, property_name, dimensions=dimensions, metric=metric)
        self._db.create_vector_index(
            label,
            property_name,
            dimensions=dimensions,
            metric=metric,
            m=m,
            ef_construction=ef_construction,
            quantization=quantization,
        )
        self._vector_metric = metric
        return None

    def vector_ann_search(
        self,
        query_vector: list[float],
        *,
        label: str = "VectorNode",
        property_name: str = "embedding",
        k: int = 10,
        ef: int | None = None,
    ) -> list[tuple[str, float]]:
        """HNSW k-NN through db.vector_search(); scores are 1 - cosine distance."""
        if not hasattr(self._db, "vector_search"):
            return super().vector_ann_search(query_vector, label=label, property_name=property_name, k=k, ef=ef)
        hits = self._db.vector_search(label, property_name, query_vector, k, ef=ef)
        app = self._app_ids(nid for nid, _ in hits)
        cosine = self._vector_metric == "cosine"
        return [(app[nid], 1.0 - distance if cosine else -distance) for nid, distance in hits]

    def delete_nodes(self, node_ids: Sequence[str]) -> int:
        """Delete nodes and their edges (GQL DETACH DELETE); returns how many were deleted."""
        result = self._db.execute(_DELETE_NODES, {"keys": list(node_ids)})
        return int(result.counters.get("nodes_deleted", 0))
