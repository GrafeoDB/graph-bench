r"""
Neo4j database adapter.

Requires: pip install neo4j

Environment variables:
    GRAPH_BENCH_NEO4J_URI: Connection URI (default: bolt://localhost:7687)
    GRAPH_BENCH_NEO4J_USER: Username (default: neo4j)
    GRAPH_BENCH_NEO4J_PASSWORD: Password (default: benchmark)

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
        password = kwargs.get("password") or get_env("NEO4J_PASSWORD", default="benchmark")

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
                query = f"UNWIND $nodes AS node CREATE (n:{label}:Node) SET n = node"
                session.run(query, nodes=batch)
                count += len(batch)
            # Create index on id for the universal :Node label to speed up label-free lookups
            try:
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.id)")
            except Exception:
                pass
        return count

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self._driver.session() as session:
            result = session.run("MATCH (n:Node {id: $id}) RETURN n", id=node_id)
            record = result.single()
            if record:
                return dict(record["n"])
            return None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n:Node {id: $id}) SET n += $props RETURN n",
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
        # Group edges by type for efficient batched UNWIND (no APOC needed)
        from collections import defaultdict

        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for src, tgt, etype, props in edges:
            by_type[etype].append({"src": src, "tgt": tgt, "props": props})

        count = 0
        with self._driver.session() as session:
            for edge_type, typed_edges in by_type.items():
                for i in range(0, len(typed_edges), batch_size):
                    batch = typed_edges[i : i + batch_size]
                    query = f"""
                    UNWIND $edges AS edge
                    MATCH (a:Node {{id: edge.src}}), (b:Node {{id: edge.tgt}})
                    CREATE (a)-[r:{edge_type}]->(b)
                    SET r = edge.props
                    """
                    session.run(query, edges=batch)
                    count += len(batch)
        return count

    def get_neighbors(self, node_id: str, *, edge_type: str | None = None) -> list[str]:
        with self._driver.session() as session:
            if edge_type:
                query = f"MATCH (n:Node {{id: $id}})-[:{edge_type}]->(m) RETURN m.id AS id"
            else:
                query = "MATCH (n:Node {id: $id})-->(m) RETURN m.id AS id"
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
                MATCH (start:Node {{id: $src}}), (end:Node {{id: $tgt}}),
                      path = shortestPath((start)-[{rel}]->(end))
                RETURN [n IN nodes(path) | n.id] AS path
                """
            else:
                query = f"""
                MATCH (start:Node {{id: $src}}), (end:Node {{id: $tgt}}),
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

    def _ensure_gds_projection(self, session: Any) -> str:
        """Create a GDS graph projection if it doesn't exist."""
        graph_name = "bench_graph"
        # Drop existing projection if any
        try:
            session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass
        # Create new projection with all nodes and relationships
        session.run(
            f"""
            CALL gds.graph.project(
                '{graph_name}',
                '*',
                '*',
                {{relationshipProperties: 'weight'}}
            )
            """
        )
        return graph_name

    def _drop_gds_projection(self, session: Any, graph_name: str) -> None:
        """Drop a GDS graph projection."""
        try:
            session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass

    def pagerank(
        self,
        *,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        """PageRank using Neo4j GDS."""
        with self._driver.session() as session:
            try:
                graph_name = self._ensure_gds_projection(session)
                result = session.run(
                    f"""
                    CALL gds.pageRank.stream('{graph_name}', {{
                        dampingFactor: $damping,
                        maxIterations: $max_iter,
                        tolerance: $tol
                    }})
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).id AS id, score
                    """,
                    damping=damping,
                    max_iter=max_iterations,
                    tol=tolerance,
                )
                scores = {record["id"]: record["score"] for record in result if record["id"]}
                self._drop_gds_projection(session, graph_name)
                if scores:
                    return scores
            except Exception:
                pass
        return super().pagerank(damping=damping, max_iterations=max_iterations, tolerance=tolerance)

    def community_detection(self, *, algorithm: str = "louvain") -> list[set[str]]:
        """Community detection using Neo4j GDS."""
        with self._driver.session() as session:
            try:
                graph_name = self._ensure_gds_projection(session)
                if algorithm == "louvain":
                    result = session.run(
                        f"""
                        CALL gds.louvain.stream('{graph_name}')
                        YIELD nodeId, communityId
                        RETURN gds.util.asNode(nodeId).id AS id, communityId
                        """
                    )
                else:  # label_propagation
                    result = session.run(
                        f"""
                        CALL gds.labelPropagation.stream('{graph_name}')
                        YIELD nodeId, communityId
                        RETURN gds.util.asNode(nodeId).id AS id, communityId
                        """
                    )
                communities: dict[int, set[str]] = {}
                for record in result:
                    cid = record["communityId"]
                    nid = record["id"]
                    if nid:
                        if cid not in communities:
                            communities[cid] = set()
                        communities[cid].add(str(nid))
                self._drop_gds_projection(session, graph_name)
                if communities:
                    return list(communities.values())
            except Exception:
                pass
        return super().community_detection(algorithm=algorithm)

    def weakly_connected_components(self) -> list[set[str]]:
        """WCC using Neo4j GDS."""
        with self._driver.session() as session:
            try:
                graph_name = self._ensure_gds_projection(session)
                result = session.run(
                    f"""
                    CALL gds.wcc.stream('{graph_name}')
                    YIELD nodeId, componentId
                    RETURN gds.util.asNode(nodeId).id AS id, componentId
                    """
                )
                components: dict[int, set[str]] = {}
                for record in result:
                    cid = record["componentId"]
                    nid = record["id"]
                    if nid:
                        if cid not in components:
                            components[cid] = set()
                        components[cid].add(str(nid))
                self._drop_gds_projection(session, graph_name)
                if components:
                    return list(components.values())
            except Exception:
                pass
        return super().weakly_connected_components()

    def local_clustering_coefficient(self) -> dict[str, float]:
        """LCC using Neo4j GDS. Requires UNDIRECTED projection."""
        with self._driver.session() as session:
            try:
                graph_name = "bench_graph_undirected"
                try:
                    session.run(f"CALL gds.graph.drop('{graph_name}', false)")
                except Exception:
                    pass
                session.run(
                    f"""
                    CALL gds.graph.project(
                        '{graph_name}',
                        '*',
                        {{__ALL__: {{type: '*', orientation: 'UNDIRECTED'}}}}
                    )
                    """
                )
                result = session.run(
                    f"""
                    CALL gds.localClusteringCoefficient.stream('{graph_name}')
                    YIELD nodeId, localClusteringCoefficient
                    RETURN gds.util.asNode(nodeId).id AS id, localClusteringCoefficient AS coeff
                    """
                )
                coeffs = {record["id"]: record["coeff"] for record in result if record["id"]}
                try:
                    session.run(f"CALL gds.graph.drop('{graph_name}', false)")
                except Exception:
                    pass
                if coeffs:
                    return coeffs
            except Exception:
                pass
        return super().local_clustering_coefficient()

    def bfs_levels(self, source: str) -> dict[str, int]:
        """BFS levels using Neo4j GDS."""
        with self._driver.session() as session:
            try:
                graph_name = self._ensure_gds_projection(session)
                # Get source node id
                source_result = session.run(
                    "MATCH (n:Node {id: $id}) RETURN id(n) AS nodeId",
                    id=source,
                )
                source_record = source_result.single()
                if not source_record:
                    self._drop_gds_projection(session, graph_name)
                    return {}
                source_node_id = source_record["nodeId"]

                result = session.run(
                    f"""
                    CALL gds.bfs.stream('{graph_name}', {{
                        sourceNode: $sourceId
                    }})
                    YIELD path
                    UNWIND nodes(path) AS node
                    WITH node, length(path) AS depth
                    RETURN node.id AS id, depth
                    """,
                    sourceId=source_node_id,
                )
                levels = {record["id"]: record["depth"] for record in result if record["id"]}
                self._drop_gds_projection(session, graph_name)
                if levels:
                    return levels
            except Exception:
                pass
        return super().bfs_levels(source)

    def sssp(self, source: str, *, weight_attr: str = "weight") -> dict[str, float]:
        """SSSP using Neo4j GDS Dijkstra."""
        with self._driver.session() as session:
            try:
                graph_name = self._ensure_gds_projection(session)
                # Get source node id
                source_result = session.run(
                    "MATCH (n:Node {id: $id}) RETURN id(n) AS nodeId",
                    id=source,
                )
                source_record = source_result.single()
                if not source_record:
                    self._drop_gds_projection(session, graph_name)
                    return {}
                source_node_id = source_record["nodeId"]

                result = session.run(
                    f"""
                    CALL gds.allShortestPaths.dijkstra.stream('{graph_name}', {{
                        sourceNode: $sourceId,
                        relationshipWeightProperty: 'weight'
                    }})
                    YIELD targetNode, totalCost
                    RETURN gds.util.asNode(targetNode).id AS id, totalCost AS distance
                    """,
                    sourceId=source_node_id,
                )
                distances = {record["id"]: record["distance"] for record in result if record["id"]}
                self._drop_gds_projection(session, graph_name)
                if distances:
                    return distances
            except Exception:
                pass
        return super().sssp(source, weight_attr=weight_attr)
