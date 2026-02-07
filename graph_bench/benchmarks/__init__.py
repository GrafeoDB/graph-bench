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
- ldbc_acid: LDBC ACID tests (atomicity, isolation anomalies G0-WS)
- ldbc_graphanalytics: LDBC Graphanalytics standard benchmarks (BFS, PR, WCC, CDLP, LCC, SSSP)
- ldbc_snb: LDBC SNB Interactive workload benchmarks (IS1, IS3, IC1, IC2, IC3, IC6)
- concurrent: Concurrent ACID tests (throughput scaling, lost updates, read-after-write)
- vector: Vector search benchmarks (insert, k-NN, batch search, recall)
- hybrid: Hybrid graph+vector benchmarks (graph-to-vector, vector-to-graph)

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
    SnbIS2Benchmark,
    SnbIS3Benchmark,
    SnbIS4Benchmark,
    SnbIS5Benchmark,
    SnbIS6Benchmark,
    SnbIS7Benchmark,
)
from graph_bench.benchmarks.concurrent import (
    ConcurrentAcidBenchmark,
    LostUpdateBenchmark,
    ReadAfterWriteBenchmark,
    ThroughputScalingBenchmark,
    MixedWorkloadBenchmark as ConcurrentMixedWorkloadBenchmark,
)
from graph_bench.benchmarks.vector import (
    VectorBatchSearchBenchmark,
    VectorInsertBenchmark,
    VectorKnnBenchmark,
    VectorRecallBenchmark,
)
from graph_bench.benchmarks.hybrid import (
    HybridGraphToVectorBenchmark,
    HybridVectorToGraphBenchmark,
)
from graph_bench.benchmarks.ldbc_acid import (
    AtomicityCommitTest,
    AtomicityRollbackTest,
    G0DirtyWriteTest,
    G1aAbortedReadTest,
    G1bIntermediateReadTest,
    G1cCircularInfoFlowTest,
    IMPItemManyPrecedersTest,
    PMPPredicateManyPrecedersTest,
    OTVObservedTxnVanishesTest,
    FRFracturedReadTest,
    LULostUpdateTest,
    WSWriteSkewTest,
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
    "SnbIS2Benchmark",
    "SnbIS3Benchmark",
    "SnbIS4Benchmark",
    "SnbIS5Benchmark",
    "SnbIS6Benchmark",
    "SnbIS7Benchmark",
    # Concurrent ACID
    "ConcurrentAcidBenchmark",
    "LostUpdateBenchmark",
    "ReadAfterWriteBenchmark",
    "ThroughputScalingBenchmark",
    "ConcurrentMixedWorkloadBenchmark",
    # Vector
    "VectorInsertBenchmark",
    "VectorKnnBenchmark",
    "VectorBatchSearchBenchmark",
    "VectorRecallBenchmark",
    # Hybrid
    "HybridGraphToVectorBenchmark",
    "HybridVectorToGraphBenchmark",
    # LDBC ACID
    "AtomicityCommitTest",
    "AtomicityRollbackTest",
    "G0DirtyWriteTest",
    "G1aAbortedReadTest",
    "G1bIntermediateReadTest",
    "G1cCircularInfoFlowTest",
    "IMPItemManyPrecedersTest",
    "PMPPredicateManyPrecedersTest",
    "OTVObservedTxnVanishesTest",
    "FRFracturedReadTest",
    "LULostUpdateTest",
    "WSWriteSkewTest",
]
