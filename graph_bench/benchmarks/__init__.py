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
- ldbc_snb: LDBC SNB Interactive workload benchmarks (IS1, IS3, IC1, IC2, IC3, IC6)
- concurrent: Concurrent ACID tests (throughput scaling, lost updates, read-after-write)

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
from graph_bench.benchmarks.snb_interactive import (
    SnbIC1Benchmark,
    SnbIC2Benchmark,
    SnbIC3Benchmark,
    SnbIC6Benchmark,
    SnbIS1Benchmark,
    SnbIS3Benchmark,
)
from graph_bench.benchmarks.concurrent import (
    ConcurrentAcidBenchmark,
    LostUpdateBenchmark,
    ReadAfterWriteBenchmark,
    ThroughputScalingBenchmark,
    MixedWorkloadBenchmark as ConcurrentMixedWorkloadBenchmark,
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
    # LDBC SNB Interactive
    "SnbIC1Benchmark",
    "SnbIC2Benchmark",
    "SnbIC3Benchmark",
    "SnbIC6Benchmark",
    "SnbIS1Benchmark",
    "SnbIS3Benchmark",
    # Concurrent ACID
    "ConcurrentAcidBenchmark",
    "LostUpdateBenchmark",
    "ReadAfterWriteBenchmark",
    "ThroughputScalingBenchmark",
    "ConcurrentMixedWorkloadBenchmark",
]
