r"""
Database adapters for graph-bench.

Each adapter implements the GraphDatabaseAdapter protocol
to provide a consistent interface across databases.

    from graph_bench.adapters import Neo4jAdapter, LadybugAdapter

    adapter = Neo4jAdapter()
    adapter.connect(uri="bolt://localhost:7687")
"""

from graph_bench.adapters.arangodb import ArangoDBAdapter
from graph_bench.adapters.base import AdapterRegistry, BaseAdapter
from graph_bench.adapters.falkordb import FalkorDBAdapter
from graph_bench.adapters.falkordblite import FalkorDBLiteAdapter
from graph_bench.adapters.grafeo import GrafeoAdapter
from graph_bench.adapters.grafeo_server import GrafeoServerAdapter
from graph_bench.adapters.ladybug import LadybugAdapter
from graph_bench.adapters.memgraph import MemgraphAdapter
from graph_bench.adapters.nebula import NebulaGraphAdapter
from graph_bench.adapters.neo4j import Neo4jAdapter
from graph_bench.adapters.tugraph import TuGraphAdapter
from graph_bench.adapters.turing import TuringDBAdapter

__all__ = [
    "AdapterRegistry",
    "ArangoDBAdapter",
    "BaseAdapter",
    "FalkorDBAdapter",
    "FalkorDBLiteAdapter",
    "GrafeoAdapter",
    "GrafeoServerAdapter",
    "LadybugAdapter",
    "MemgraphAdapter",
    "NebulaGraphAdapter",
    "Neo4jAdapter",
    "TuGraphAdapter",
    "TuringDBAdapter",
]
