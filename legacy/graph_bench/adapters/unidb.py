r"""
Uni-DB database adapter.

Uni-DB is an embedded multi-model database (graph + vector + columnar)
with OpenCypher queries and 36 native graph algorithms.

Requires: pip install uni-db

Environment variables:
    GRAPH_BENCH_UNIDB_PATH: Database path (default: in-memory)

    from graph_bench.adapters.unidb import UniDBAdapter

    adapter = UniDBAdapter()
    adapter.connect()  # In-memory by default
"""

from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["UniDBAdapter"]


@AdapterRegistry.register("unidb")
class UniDBAdapter(BaseAdapter):
    """Uni-DB embedded multi-model database adapter."""

    def __init__(self) -> None:
        self._uni: Any = None
        self._session: Any = None
        self._connected = False
        self._schema_ready = False

    @property
    def name(self) -> str:
        return "Uni-DB"

    @property
    def version(self) -> str:
        return "1.0.2"

    @property
    def is_embedded(self) -> bool:
        return True

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            import uni_db  # noqa: F401
        except ImportError as e:
            msg = "uni-db package not installed. Install with: pip install uni-db"
            raise ImportError(msg) from e

        import uni_db

        path = uri or kwargs.get("path") or get_env("UNIDB_PATH")

        if not path or path == ":memory:":
            self._uni = uni_db.Uni.in_memory()
        else:
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)
            self._uni = uni_db.Uni.open(path)

        self._session = self._uni.session()
        self._connected = True
        self._schema_ready = False

    def _ensure_schema(self) -> None:
        """Create node and edge schema if not already done."""
        if self._schema_ready:
            return
        try:
            tx = self._session.tx()
            tx.execute(
                "CREATE LABEL Node ("
                "id STRING NOT NULL, "
                "label STRING, "
                "name STRING, "
                "age INT64, "
                "city STRING, "
                "score INT64, "
                "firstName STRING, "
                "lastName STRING"
                ")"
            )
            tx.execute(
                "CREATE EDGE TYPE EDGE ("
                "edge_type STRING, "
                "weight FLOAT64"
                ") FROM Node TO Node"
            )
            tx.commit()
            self._schema_ready = True
        except Exception:
            # Schema may already exist
            self._schema_ready = True

    def disconnect(self) -> None:
        if self._uni:
            try:
                self._uni.shutdown()
            except Exception:
                pass
        self._uni = None
        self._session = None
        self._connected = False
        self._schema_ready = False

    def clear(self) -> None:
        """Remove all data by recreating the database."""
        import uni_db
        if self._uni:
            try:
                self._uni.shutdown()
            except Exception:
                pass
        self._uni = uni_db.Uni.in_memory()
        self._session = self._uni.session()
        self._schema_ready = False

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        self._ensure_schema()
        if not nodes:
            return 0

        # Known columns in schema
        known = {"id", "label", "name", "age", "city", "score",
                 "firstName", "lastName"}

        rows = []
        for i, node in enumerate(nodes):
            row: dict[str, Any] = {"id": str(node.get("id", i))}
            for k in ("label", "name", "city", "firstName", "lastName"):
                if k in node:
                    row[k] = str(node[k]) if node[k] is not None else None
            for k in ("age", "score"):
                if k in node:
                    row[k] = int(node[k]) if node[k] is not None else None
            # Set label if not present
            if "label" not in row:
                row["label"] = label
            rows.append(row)

        count = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            try:
                ws = self._uni.session()
                tx = ws.tx()
                bw = tx.bulk_writer().build()
                bw.insert_vertices("Node", batch)
                bw.commit()
                count += len(batch)
            except Exception:
                # Fallback to individual inserts
                for row in batch:
                    try:
                        ws2 = self._uni.session()
                        tx2 = ws2.tx()
                        parts = ", ".join(
                            f"{k}: ${k}" for k in row if row[k] is not None
                        )
                        params = {
                            k: v for k, v in row.items() if v is not None
                        }
                        tx2.execute(
                            f"CREATE (:Node {{{parts}}})", params
                        )
                        tx2.commit()
                        count += 1
                    except Exception:
                        pass

        self._session.refresh()
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        self._ensure_schema()
        try:
            self._session.refresh()
            result = self._session.query(
                "MATCH (n:Node {id: $id}) RETURN n.id, n.name, n.age, "
                "n.city, n.label, n.score, n.firstName, n.lastName",
                {"id": node_id},
            )
            if result.rows:
                d = result.rows[0].to_dict()
                node: dict[str, Any] = {}
                col_map = {
                    "n.id": "id", "n.name": "name", "n.age": "age",
                    "n.city": "city", "n.label": "label", "n.score": "score",
                    "n.firstName": "firstName", "n.lastName": "lastName",
                }
                for col, key in col_map.items():
                    if col in d and d[col] is not None:
                        node[key] = d[col]
                return node if node else None
        except Exception:
            pass
        return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        self._ensure_schema()
        try:
            set_parts = ", ".join(f"n.{k} = ${k}" for k in properties)
            params = {"id": node_id, **properties}
            ws = self._uni.session()
            tx = ws.tx()
            tx.execute(
                f"MATCH (n:Node {{id: $id}}) SET {set_parts}",
                params,
            )
            tx.commit()
            self._session.refresh()
            return True
        except Exception:
            return False

    def get_nodes_by_label(
        self, label: str, *, limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._ensure_schema()
        nodes = []
        try:
            self._session.refresh()
            result = self._session.query(
                "MATCH (n:Node) WHERE n.label = $label "
                "RETURN n.id, n.name, n.age, n.city, n.label, n.score, "
                "n.firstName, n.lastName "
                f"LIMIT {limit}",
                {"label": label},
            )
            col_map = {
                "n.id": "id", "n.name": "name", "n.age": "age",
                "n.city": "city", "n.label": "label", "n.score": "score",
                "n.firstName": "firstName", "n.lastName": "lastName",
            }
            for row in result.rows:
                d = row.to_dict()
                node = {
                    key: d[col] for col, key in col_map.items()
                    if col in d and d[col] is not None
                }
                if node:
                    nodes.append(node)
        except Exception:
            pass
        return nodes

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        self._ensure_schema()
        if not edges:
            return 0

        # Build id -> vid mapping
        self._session.refresh()
        result = self._session.query(
            "MATCH (n:Node) RETURN n.id AS nid, id(n) AS vid"
        )
        id_to_vid: dict[str, Any] = {}
        for row in result.rows:
            d = row.to_dict()
            id_to_vid[str(d["nid"])] = d["vid"]

        count = 0
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            edge_data = []
            for src, tgt, edge_type, props in batch:
                src_vid = id_to_vid.get(src)
                tgt_vid = id_to_vid.get(tgt)
                if src_vid is not None and tgt_vid is not None:
                    edge_data.append((
                        src_vid,
                        tgt_vid,
                        {
                            "edge_type": edge_type,
                            "weight": float(props.get("weight", 0.0)),
                        },
                    ))
            if edge_data:
                try:
                    ws = self._uni.session()
                    tx = ws.tx()
                    bw = tx.bulk_writer().build()
                    bw.insert_edges("EDGE", edge_data)
                    bw.commit()
                    count += len(edge_data)
                except Exception:
                    # Fallback to individual inserts
                    for src, tgt, edge_type, props in batch:
                        try:
                            ws2 = self._uni.session()
                            tx2 = ws2.tx()
                            tx2.execute(
                                "MATCH (a:Node {id: $src}), (b:Node {id: $tgt}) "
                                "CREATE (a)-[:EDGE {edge_type: $type, weight: $w}]->(b)",
                                {
                                    "src": src,
                                    "tgt": tgt,
                                    "type": edge_type,
                                    "w": float(props.get("weight", 0.0)),
                                },
                            )
                            tx2.commit()
                            count += 1
                        except Exception:
                            pass

        self._session.refresh()
        return count

    def get_neighbors(
        self, node_id: str, *, edge_type: str | None = None,
    ) -> list[str]:
        self._ensure_schema()
        neighbors = []
        try:
            self._session.refresh()
            if edge_type:
                result = self._session.query(
                    "MATCH (n:Node {id: $id})-[e:EDGE]->(m:Node) "
                    "WHERE e.edge_type = $type RETURN m.id AS mid",
                    {"id": node_id, "type": edge_type},
                )
            else:
                result = self._session.query(
                    "MATCH (n:Node {id: $id})-[:EDGE]->(m:Node) "
                    "RETURN m.id AS mid",
                    {"id": node_id},
                )
            for row in result.rows:
                d = row.to_dict()
                if d.get("mid") is not None:
                    neighbors.append(str(d["mid"]))
        except Exception:
            pass
        return neighbors

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        from collections import deque

        visited: set[str] = set()
        queue: deque[tuple[str, list[str]]] = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.get_neighbors(current, edge_type=edge_type):
                if neighbor not in visited:
                    queue.append((neighbor, [*path, neighbor]))

        return None

    def execute_query(
        self, query: str, *, params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_schema()
        results = []
        try:
            self._session.refresh()
            result = self._session.query(query, params or {})
            for row in result.rows:
                results.append(row.to_dict())
        except Exception:
            pass
        return results

    def count_nodes(self, *, label: str | None = None) -> int:
        self._ensure_schema()
        try:
            self._session.refresh()
            if label:
                result = self._session.query(
                    "MATCH (n:Node) WHERE n.label = $label "
                    "RETURN count(n) AS cnt",
                    {"label": label},
                )
            else:
                result = self._session.query(
                    "MATCH (n:Node) RETURN count(n) AS cnt"
                )
            if result.rows:
                return result.rows[0].to_dict()["cnt"]
        except Exception:
            pass
        return 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        self._ensure_schema()
        try:
            self._session.refresh()
            if edge_type:
                result = self._session.query(
                    "MATCH ()-[e:EDGE]->() WHERE e.edge_type = $type "
                    "RETURN count(e) AS cnt",
                    {"type": edge_type},
                )
            else:
                result = self._session.query(
                    "MATCH ()-[e:EDGE]->() RETURN count(e) AS cnt"
                )
            if result.rows:
                return result.rows[0].to_dict()["cnt"]
        except Exception:
            pass
        return 0

    # --- Native graph algorithms ---

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        self._session.refresh()
        result = self._session.query(
            "CALL uni.algo.pageRank(['Node'], ['EDGE']) "
            "YIELD nodeId, score RETURN nodeId, score"
        )
        return {
            str(row.to_dict()["nodeId"]): float(row.to_dict()["score"])
            for row in result.rows
        }

    def weakly_connected_components(self) -> list[set[str]]:
        self._session.refresh()
        result = self._session.query(
            "CALL uni.algo.wcc(['Node'], ['EDGE']) "
            "YIELD nodeId, componentId RETURN nodeId, componentId"
        )
        components: dict[int, set[str]] = {}
        for row in result.rows:
            d = row.to_dict()
            cid = int(d["componentId"])
            if cid not in components:
                components[cid] = set()
            components[cid].add(str(d["nodeId"]))
        return list(components.values())

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        self._session.refresh()
        if algorithm == "louvain":
            result = self._session.query(
                "CALL uni.algo.louvain(['Node'], ['EDGE']) "
                "YIELD nodeId, communityId RETURN nodeId, communityId"
            )
        else:
            result = self._session.query(
                "CALL uni.algo.labelPropagation(['Node'], ['EDGE']) "
                "YIELD nodeId, communityId RETURN nodeId, communityId"
            )
        communities: dict[int, set[str]] = {}
        for row in result.rows:
            d = row.to_dict()
            cid = int(d["communityId"])
            if cid not in communities:
                communities[cid] = set()
            communities[cid].add(str(d["nodeId"]))
        return list(communities.values())
