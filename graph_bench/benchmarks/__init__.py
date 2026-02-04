r"""
Benchmark implementations for graph-bench.

Benchmarks are organized by category:
- storage: Node/edge insertion, read, update, delete
- traversal: 1-hop, 2-hop, k-hop, pattern matching
- algorithms: BFS, DFS, PageRank, shortest paths, centrality
- query: Aggregations, filtering
- pattern: Triangle counting, common neighbors
- structure: Connected components, degree distribution
- write: Property updates, mixed workloads
- ldbc_graphanalytics: LDBC Graphanalytics standard benchmarks (BFS, PR, WCC, CDLP, LCC, SSSP)

    from graph_bench.benchmarks import NodeInsertionBenchmark, BFSBenchmark
"""

from graph_bench.benchmarks.algorithms import (
    BetweennessCentralityBenchmark,
    ClosenessCentralityBenchmark,
    CommunityDetectionBenchmark,
    PageRankBenchmark,
)
from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.benchmarks.pattern import (
    CommonNeighborsBenchmark,
    TriangleCountBenchmark,
)
from graph_bench.benchmarks.query import AggregationCountBenchmark, FilterEqualityBenchmark, FilterRangeBenchmark
from graph_bench.benchmarks.storage import (
    BatchReadBenchmark,
    EdgeInsertionBenchmark,
    NodeInsertionBenchmark,
    SingleReadBenchmark,
)
from graph_bench.benchmarks.structure import (
    ConnectedComponentsBenchmark,
    DegreeDistributionBenchmark,
    GraphDensityBenchmark,
    ReachabilityBenchmark,
)
from graph_bench.benchmarks.traversal import (
    BFSBenchmark,
    DFSBenchmark,
    Hop1Benchmark,
    Hop2Benchmark,
    ShortestPathBenchmark,
)
from graph_bench.benchmarks.write import (
    EdgeAddExistingBenchmark,
    MixedWorkloadBenchmark,
    PropertyUpdateBenchmark,
)
from graph_bench.benchmarks.graphanalytics import (
    LdbcBfsBenchmark,
    LdbcCdlpBenchmark,
    LdbcLccBenchmark,
    LdbcPageRankBenchmark,
    LdbcSsspBenchmark,
    LdbcWccBenchmark,
)

__all__ = [
    # Base
    "BaseBenchmark",
    "BenchmarkRegistry",
    # Storage
    "BatchReadBenchmark",
    "EdgeInsertionBenchmark",
    "NodeInsertionBenchmark",
    "SingleReadBenchmark",
    # Traversal
    "BFSBenchmark",
    "DFSBenchmark",
    "Hop1Benchmark",
    "Hop2Benchmark",
    "ShortestPathBenchmark",
    # Algorithms
    "BetweennessCentralityBenchmark",
    "ClosenessCentralityBenchmark",
    "CommunityDetectionBenchmark",
    "PageRankBenchmark",
    # Query
    "AggregationCountBenchmark",
    "FilterEqualityBenchmark",
    "FilterRangeBenchmark",
    # Pattern
    "CommonNeighborsBenchmark",
    "TriangleCountBenchmark",
    # Structure
    "ConnectedComponentsBenchmark",
    "DegreeDistributionBenchmark",
    "GraphDensityBenchmark",
    "ReachabilityBenchmark",
    # Write
    "EdgeAddExistingBenchmark",
    "MixedWorkloadBenchmark",
    "PropertyUpdateBenchmark",
    # LDBC Graphanalytics
    "LdbcBfsBenchmark",
    "LdbcCdlpBenchmark",
    "LdbcLccBenchmark",
    "LdbcPageRankBenchmark",
    "LdbcSsspBenchmark",
    "LdbcWccBenchmark",
]
