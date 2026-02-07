r"""
Memgraph database adapter.

Memgraph uses the Bolt protocol, same as Neo4j.
Requires: pip install neo4j

Environment variables:
    GRAPH_BENCH_MEMGRAPH_URI: Connection URI (default: bolt://localhost:7688)

    from graph_bench.adapters.memgraph import MemgraphAdapter

    adapter = MemgraphAdapter()
    adapter.connect(uri="bolt://localhost:7688")
"""

from collections.abc import Sequence
from typing import Any

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.config import get_env

__all__ = ["MemgraphAdapter"]


@AdapterRegistry.register("memgraph")
class MemgraphAdapter(BaseAdapter):
    """Memgraph graph database adapter."""

    def __init__(self) -> None:
        self._driver: Any = None
        self._connected = False

    @property
    def name(self) -> str:
        return "Memgraph"

    @property
    def version(self) -> str:
        if not self._connected or self._driver is None:
            return "unknown"
        try:
            with self._driver.session() as session:
                result = session.run("CALL mg.info() YIELD value RETURN value")
                for record in result:
                    val = record["value"]
                    if isinstance(val, dict) and "version" in val:
                        return val["version"]
                return "unknown"
        except Exception:
            return "unknown"

    def connect(self, *, uri: str | None = None, **kwargs: Any) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            msg = "neo4j package not installed. Install with: pip install neo4j"
            raise ImportError(msg) from e

        uri = uri or get_env("MEMGRAPH_URI", default="bolt://localhost:7688")

        if uri is None:
            msg = "Memgraph URI required"
            raise ValueError(msg)

        self._driver = GraphDatabase.driver(uri)
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
            # Create index on id for this label to speed up MATCH in insert_edges
            try:
                session.run(f"CREATE INDEX ON :{label}(id)")
            except Exception:
                pass  # Index may already exist
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._driver.session() as session:
            result = session.run("MATCH (n {id: $id}) RETURN n", id=node_id)
            record = result.single()
            if record:
                return dict(record["n"])
            return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n {id: $id}) SET n += $props RETURN n",
                id=node_id,
                props=properties,
            )
            return result.single() is not None

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
                batch = edges[i : i + batch_size]
                # Group edges by type for UNWIND (Cypher requires static rel types)
                by_type: dict[str, list[dict[str, Any]]] = {}
                for src, tgt, edge_type, props in batch:
                    if edge_type not in by_type:
                        by_type[edge_type] = []
                    by_type[edge_type].append({"src": src, "tgt": tgt, "props": props})

                for edge_type, edge_list in by_type.items():
                    try:
                        query = f"""
                        UNWIND $edges AS e
                        MATCH (a {{id: e.src}}), (b {{id: e.tgt}})
                        CREATE (a)-[r:{edge_type}]->(b)
                        SET r = e.props
                        """
                        session.run(query, edges=edge_list)
                        count += len(edge_list)
                    except Exception:
                        # Fall back to individual inserts
                        for edge in edge_list:
                            query = f"""
                            MATCH (a {{id: $src}}), (b {{id: $tgt}})
                            CREATE (a)-[r:{edge_type}]->(b)
                            SET r = $props
                            """
                            session.run(
                                query,
                                src=edge["src"],
                                tgt=edge["tgt"],
                                props=edge["props"],
                            )
                            count += 1
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
        # Memgraph uses variable-length path with BFS (default behavior)
        with self._driver.session() as session:
            if edge_type:
                rel = f":{edge_type}*1..15"
            else:
                rel = "*1..15"

            # Variable-length path finds shortest by default in Memgraph
            query = f"""
            MATCH (start {{id: $src}}), (end {{id: $tgt}})
            MATCH path = (start)-[{rel}]->(end)
            RETURN [n IN nodes(path) | n.id] AS path
            ORDER BY length(path)
            LIMIT 1
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
        """PageRank using MAGE pagerank.get()."""
        # MAGE pagerank.get() signature:
        # pagerank.get(max_iterations, damping_factor, stop_epsilon)
        query = """
        CALL pagerank.get($max_iterations, $damping_factor, $stop_epsilon)
        YIELD node, rank
        RETURN node.id AS id, rank AS score
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    query,
                    max_iterations=max_iterations,
                    damping_factor=damping,
                    stop_epsilon=tolerance,
                )
                return {
                    record["id"]: record["score"]
                    for record in result
                    if record["id"]
                }
        except Exception as e:
            msg = f"Memgraph PageRank failed. Ensure MAGE is installed: {e}"
            raise NotImplementedError(msg) from e

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Community detection using MAGE."""
        if algorithm == "louvain":
            query = """
            CALL community_detection.get()
            YIELD node, community_id
            RETURN node.id AS id, community_id
            """
        else:  # label_propagation -> use leiden
            query = """
            CALL leiden_community_detection.get()
            YIELD node, community_id
            RETURN node.id AS id, community_id
            """
        try:
            with self._driver.session() as session:
                result = session.run(query)
                communities: dict[int, set[str]] = {}
                for record in result:
                    cid = record["community_id"]
                    if cid not in communities:
                        communities[cid] = set()
                    communities[cid].add(record["id"])
                return list(communities.values())
        except Exception as e:
            msg = f"Memgraph community detection failed: {e}"
            raise NotImplementedError(msg) from e

    def weakly_connected_components(self) -> list[set[str]]:
        """WCC using MAGE."""
        query = """
        CALL weakly_connected_components.get()
        YIELD node, component_id
        RETURN node.id AS id, component_id
        """
        try:
            with self._driver.session() as session:
                result = session.run(query)
                components: dict[int, set[str]] = {}
                for record in result:
                    cid = record["component_id"]
                    if cid not in components:
                        components[cid] = set()
                    components[cid].add(record["id"])
                return list(components.values())
        except Exception as e:
            msg = f"Memgraph WCC failed. Ensure MAGE is installed: {e}"
            raise NotImplementedError(msg) from e

    def local_clustering_coefficient(self) -> dict[str, float]:
        """LCC using MAGE nxalg.clustering()."""
        # nxalg.clustering() computes clustering coefficient for all nodes
        query = """
        CALL nxalg.clustering()
        YIELD node, clustering
        RETURN node.id AS id, clustering AS coeff
        """
        try:
            with self._driver.session() as session:
                result = session.run(query)
                return {
                    record["id"]: record["coeff"]
                    for record in result
                    if record["id"]
                }
        except Exception as e:
            msg = f"Memgraph LCC failed. Ensure MAGE is installed: {e}"
            raise NotImplementedError(msg) from e

    def bfs_levels(self, source: str) -> dict[str, int]:
        """BFS levels using pure Cypher variable-length paths."""
        # Use Cypher variable-length path to find distances from source
        query = """
        MATCH (start {id: $source})
        CALL {
            WITH start
            MATCH path = (start)-[*0..20]-(node)
            WITH node, min(length(path)) AS depth
            RETURN node, depth
        }
        RETURN node.id AS id, depth
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, source=source)
                return {
                    record["id"]: record["depth"]
                    for record in result
                    if record["id"]
                }
        except Exception as e:
            msg = f"Memgraph BFS failed: {e}"
            raise NotImplementedError(msg) from e

    def sssp(self, source: str, *, weight_attr: str = "weight") -> dict[str, float]:
        """SSSP using MAGE nxalg.all_pairs_dijkstra_path_length or fallback."""
        # Try multi_source_dijkstra_path_length with single source
        query = """
        MATCH (start {id: $source})
        WITH [start] AS sources
        CALL nxalg.multi_source_dijkstra_path_length(sources, null, $weight_attr)
        YIELD target, length
        RETURN target.id AS id, length AS distance
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, source=source, weight_attr=weight_attr)
                return {
                    record["id"]: record["distance"]
                    for record in result
                    if record["id"]
                }
        except Exception:
            # Fall back to NetworkX implementation from base class
            return super().sssp(source, weight_attr=weight_attr)
