r"""
Neo4j database adapter.

Requires: pip install neo4j

Environment variables:
    GRAPH_BENCH_NEO4J_URI: Connection URI (default: bolt://localhost:7687)
    GRAPH_BENCH_NEO4J_USER: Username (default: neo4j)
    GRAPH_BENCH_NEO4J_PASSWORD: Password

    from graph_bench.adapters.neo4j import Neo4jAdapter

    adapter = Neo4jAdapter()
    adapter.connect(uri="bolt://localhost:7687", user="neo4j", password="password")
"""

from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["Neo4jAdapter"]


@AdapterRegistry.register("neo4j")
class Neo4jAdapter(BaseAdapter):
    """Neo4j graph database adapter."""

    def __init__(self) -> None:
        self._driver: Any = None
        self._connected = False

    @property
    def name(self) -> str:
        return "Neo4j"

    @property
    def version(self) -> str:
        if not self._connected or self._driver is None:
            return "unknown"
        try:
            with self._driver.session() as session:
                result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] as version")
                record = result.single()
                return record["version"] if record else "unknown"
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            msg = "neo4j package not installed. Install with: pip install neo4j"
            raise ImportError(msg) from e

        uri = uri or get_env("NEO4J_URI", default="bolt://localhost:7687")
        user = kwargs.get("user") or get_env("NEO4J_USER", default="neo4j")
        password = kwargs.get("password") or get_env("NEO4J_PASSWORD")

        if uri is None:
            msg = "Neo4j URI required"
            raise ValueError(msg)

        auth = (user, password) if password else None
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._driver.verify_connectivity()
        self._connected = True

    def disconnect(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
        self._connected = False

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def insert_nodes(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        label: str = "Node",
        batch_size: int = 1000,
    ) -> int:
        count = 0
        with self._driver.session() as session:
            for i in range(0, len(nodes), batch_size):
                batch = list(nodes[i : i + batch_size])
                query = f"UNWIND $nodes AS node CREATE (n:{label}) SET n = node"
                session.run(query, nodes=batch)
                count += len(batch)
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._driver.session() as session:
            result = session.run("MATCH (n {id: $id}) RETURN n", id=node_id)
            record = result.single()
            if record:
                return dict(record["n"])
            return None

    def get_nodes_by_label(self, label: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(f"MATCH (n:{label}) RETURN n LIMIT $limit", limit=limit)
            return [dict(record["n"]) for record in result]

    def insert_edges(
        self,
        edges: Sequence[tuple[str, str, str, dict[str, Any]]],
        *,
        batch_size: int = 1000,
    ) -> int:
        count = 0
        with self._driver.session() as session:
            for i in range(0, len(edges), batch_size):
                batch = [{"src": e[0], "tgt": e[1], "type": e[2], "props": e[3]} for e in edges[i : i + batch_size]]
                query = """
                UNWIND $edges AS edge
                MATCH (a {id: edge.src}), (b {id: edge.tgt})
                CALL apoc.create.relationship(a, edge.type, edge.props, b) YIELD rel
                RETURN count(rel)
                """
                try:
                    session.run(query, edges=batch)
                except Exception:
                    for edge in batch:
                        edge_type = edge["type"]
                        q = f"""
                        MATCH (a {{id: $src}}), (b {{id: $tgt}})
                        CREATE (a)-[r:{edge_type}]->(b)
                        SET r = $props
                        """
                        session.run(q, src=edge["src"], tgt=edge["tgt"], props=edge["props"])
                count += len(batch)
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        with self._driver.session() as session:
            if edge_type:
                query = f"MATCH (n {{id: $id}})-[:{edge_type}]->(m) RETURN m.id AS id"
            else:
                query = "MATCH (n {id: $id})-->(m) RETURN m.id AS id"
            result = session.run(query, id=node_id)
            return [record["id"] for record in result if record["id"]]

    def shortest_path(
        self,
        source: str,
        target: str,
        *,
        edge_type: str | None = None,
        weighted: bool = False,
    ) -> list[str] | None:
        with self._driver.session() as session:
            if edge_type:
                rel = f":{edge_type}*"
            else:
                rel = "*"

            if weighted:
                query = f"""
                MATCH (start {{id: $src}}), (end {{id: $tgt}}),
                      path = shortestPath((start)-[{rel}]->(end))
                RETURN [n IN nodes(path) | n.id] AS path
                """
            else:
                query = f"""
                MATCH (start {{id: $src}}), (end {{id: $tgt}}),
                      path = shortestPath((start)-[{rel}]->(end))
                RETURN [n IN nodes(path) | n.id] AS path
                """

            result = session.run(query, src=source, tgt=target)
            record = result.single()
            if record and record["path"]:
                return [str(n) for n in record["path"]]
            return None

    def execute_query(self, query: str, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]

    def count_nodes(self, *, label: str | None = None) -> int:
        with self._driver.session() as session:
            if label:
                query = f"MATCH (n:{label}) RETURN count(n) AS count"
            else:
                query = "MATCH (n) RETURN count(n) AS count"
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0

    def count_edges(self, *, edge_type: str | None = None) -> int:
        with self._driver.session() as session:
            if edge_type:
                query = f"MATCH ()-[r:{edge_type}]->() RETURN count(r) AS count"
            else:
                query = "MATCH ()-[r]->() RETURN count(r) AS count"
            result = session.run(query)
            record = result.single()
            return record["count"] if record else 0

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        # Requires Neo4j Graph Data Science (GDS) plugin - not in Community Edition
        msg = (
            "Neo4j PageRank requires the Graph Data Science (GDS) plugin. "
            "Community Edition does not include GDS."
        )
        raise NotImplementedError(msg)

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        # Requires Neo4j Graph Data Science (GDS) plugin - not in Community Edition
        msg = (
            "Neo4j community detection requires the Graph Data Science (GDS) plugin. "
            "Community Edition does not include GDS."
        )
        raise NotImplementedError(msg)
