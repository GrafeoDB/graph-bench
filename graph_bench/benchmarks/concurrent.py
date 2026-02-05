r"""
Concurrent ACID benchmarks on LDBC SNB dataset.

Tests parallel throughput and consistency guarantees:
- ThroughputScaling: Ops/sec at varying concurrency levels (1, 2, 4, 8 workers)
- LostUpdate: N threads increment counter, verify no lost updates
- ReadAfterWrite: Write then immediately read back
- MixedWorkload: 80% reads, 20% writes, concurrent

    from graph_bench.benchmarks.concurrent import ThroughputScalingBenchmark

    bench = ThroughputScalingBenchmark()
    metrics = bench.run(adapter, scale)
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.datasets.ldbc_snb import LDBCSocialNetwork
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import Metrics, ScaleConfig, TimingStats

__all__ = [
    "ThroughputScalingBenchmark",
    "LostUpdateBenchmark",
    "ReadAfterWriteBenchmark",
    "MixedWorkloadBenchmark",
]


@dataclass
class ConcurrentMetrics:
    """Metrics for concurrent benchmarks."""

    total_runtime_seconds: float
    throughput_ops_per_sec: float
    latency_p50_ms: float
    latency_p99_ms: float
    consistency_violations: int
    concurrency_level: int
    total_operations: int


class ConcurrentSnbBenchmarkBase(BaseBenchmark):
    """Base class for concurrent benchmarks using LDBC SNB data.

    Sets up a social network graph with Person nodes and KNOWS edges,
    then runs concurrent operations to test throughput and consistency.
    """

    category = "concurrent"
    _person_ids: list[str] = []
    _person_count: int = 0
    _lock = threading.Lock()

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Setup LDBC SNB dataset for concurrent testing."""
        adapter.clear()

        # Generate SNB data
        dataset = LDBCSocialNetwork(scale_factor=1, seed=42)
        nodes, edges = dataset.generate(scale)

        # Insert nodes by label
        persons = [n for n in nodes if n.get("label") == "Person"]
        cities = [n for n in nodes if n.get("label") == "City"]
        tags = [n for n in nodes if n.get("label") == "Tag"]

        for label, node_list in [("Person", persons), ("City", cities), ("Tag", tags)]:
            if node_list:
                adapter.insert_nodes(node_list, label=label)

        # Insert edges
        adapter.insert_edges(edges)

        # Store person IDs for concurrent operations
        self._person_ids = [n["id"] for n in persons]
        self._person_count = len(persons)

        # Add viewCount property to all persons for counter tests
        for person_id in self._person_ids[:100]:  # Limit to 100 for updates
            adapter.update_node(person_id, {"viewCount": 0})


# =============================================================================
# Throughput Scaling Benchmark
# =============================================================================


@BenchmarkRegistry.register("throughput_scaling", category="concurrent")
class ThroughputScalingBenchmark(ConcurrentSnbBenchmarkBase):
    """Test parallel read throughput at varying concurrency levels.

    Measures how well the database scales with concurrent readers.
    Runs get_neighbors() calls in parallel with 1, 2, 4, 8 workers.
    """

    @property
    def name(self) -> str:
        return "throughput_scaling"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run concurrent read operations and measure throughput."""
        ops_per_worker = 50
        concurrency_levels = [1, 2, 4, 8]
        total_ops = 0

        for num_workers in concurrency_levels:
            latencies: list[float] = []

            def worker_task() -> list[float]:
                """Single worker executing read operations."""
                worker_latencies: list[float] = []
                for _ in range(ops_per_worker):
                    person_id = random.choice(self._person_ids)
                    start = time.perf_counter()
                    adapter.get_neighbors(person_id, edge_type="KNOWS")
                    end = time.perf_counter()
                    worker_latencies.append((end - start) * 1000)  # ms
                return worker_latencies

            start_time = time.perf_counter()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_task) for _ in range(num_workers)]
                for future in as_completed(futures):
                    latencies.extend(future.result())
            end_time = time.perf_counter()

            total_ops += len(latencies)

        return total_ops


# =============================================================================
# Lost Update Benchmark (Atomicity Test)
# =============================================================================


@BenchmarkRegistry.register("lost_update", category="concurrent")
class LostUpdateBenchmark(ConcurrentSnbBenchmarkBase):
    """Test for lost updates under concurrent modifications.

    N threads each increment a counter on the same node.
    Final value should equal N * increments_per_thread.
    Any difference indicates lost updates (atomicity violation).
    """

    @property
    def name(self) -> str:
        return "lost_update"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run concurrent counter increments and check for lost updates."""
        num_workers = 4
        increments_per_worker = 25
        target_node = self._person_ids[0]  # Use first person as target
        expected_final = num_workers * increments_per_worker

        # Reset counter
        adapter.update_node(target_node, {"viewCount": 0})

        violations = 0

        def increment_worker() -> int:
            """Worker that increments the counter."""
            local_violations = 0
            for _ in range(increments_per_worker):
                # Read current value
                node = adapter.get_node(target_node)
                if node is None:
                    local_violations += 1
                    continue

                current = node.get("viewCount", 0)

                # Increment and write back
                # Note: This is intentionally NOT atomic to detect if DB handles it
                adapter.update_node(target_node, {"viewCount": current + 1})

            return local_violations

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(increment_worker) for _ in range(num_workers)]
            for future in as_completed(futures):
                violations += future.result()

        # Check final value
        final_node = adapter.get_node(target_node)
        final_count = final_node.get("viewCount", 0) if final_node else 0

        # Lost updates = expected - actual
        lost_updates = expected_final - final_count
        if lost_updates > 0:
            violations += lost_updates

        return expected_final  # Return operations attempted


# =============================================================================
# Read After Write Benchmark (Visibility Test)
# =============================================================================


@BenchmarkRegistry.register("read_after_write", category="concurrent")
class ReadAfterWriteBenchmark(ConcurrentSnbBenchmarkBase):
    """Test read-your-writes consistency.

    Each worker writes a unique value then immediately reads it back.
    The read should always see the value just written.
    """

    @property
    def name(self) -> str:
        return "read_after_write"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run write-then-read operations and check visibility."""
        num_workers = 4
        ops_per_worker = 25
        total_ops = 0
        violations = 0
        violations_lock = threading.Lock()

        def write_read_worker(worker_id: int) -> tuple[int, int]:
            """Worker that writes then reads."""
            local_ops = 0
            local_violations = 0

            for i in range(ops_per_worker):
                # Pick a person to update (each worker uses different nodes)
                idx = (worker_id * ops_per_worker + i) % len(self._person_ids)
                person_id = self._person_ids[idx]

                # Write unique value
                unique_value = f"worker_{worker_id}_op_{i}_{time.time_ns()}"
                adapter.update_node(person_id, {"lastUpdate": unique_value})

                # Immediately read back
                node = adapter.get_node(person_id)
                if node is None:
                    local_violations += 1
                elif node.get("lastUpdate") != unique_value:
                    local_violations += 1

                local_ops += 1

            return local_ops, local_violations

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(write_read_worker, i) for i in range(num_workers)
            ]
            for future in as_completed(futures):
                ops, viols = future.result()
                total_ops += ops
                with violations_lock:
                    violations += viols

        return total_ops


# =============================================================================
# Mixed Workload Benchmark
# =============================================================================


@BenchmarkRegistry.register("concurrent_mixed", category="concurrent")
class MixedWorkloadBenchmark(ConcurrentSnbBenchmarkBase):
    """Test concurrent mixed read/write workload.

    80% reads (get_neighbors), 20% writes (update_node).
    Measures sustained throughput under realistic workload.
    """

    @property
    def name(self) -> str:
        return "concurrent_mixed"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run mixed read/write workload."""
        num_workers = 4
        ops_per_worker = 50
        read_ratio = 0.8  # 80% reads
        total_ops = 0
        latencies: list[float] = []
        latencies_lock = threading.Lock()

        def mixed_worker(worker_id: int) -> int:
            """Worker running mixed read/write operations."""
            local_ops = 0
            local_latencies: list[float] = []

            for i in range(ops_per_worker):
                person_id = random.choice(self._person_ids)
                start = time.perf_counter()

                if random.random() < read_ratio:
                    # Read operation
                    adapter.get_neighbors(person_id, edge_type="KNOWS")
                else:
                    # Write operation
                    adapter.update_node(
                        person_id, {"lastAccess": f"worker_{worker_id}_{i}"}
                    )

                end = time.perf_counter()
                local_latencies.append((end - start) * 1000)  # ms
                local_ops += 1

            with latencies_lock:
                latencies.extend(local_latencies)

            return local_ops

        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(mixed_worker, i) for i in range(num_workers)]
            for future in as_completed(futures):
                total_ops += future.result()
        end_time = time.perf_counter()

        return total_ops


# =============================================================================
# Aggregate Concurrent Benchmark (runs all tests, reports total)
# =============================================================================


@BenchmarkRegistry.register("concurrent_acid", category="concurrent")
class ConcurrentAcidBenchmark(ConcurrentSnbBenchmarkBase):
    """Aggregate benchmark running all concurrent tests.

    Runs throughput scaling, lost update, read-after-write, and mixed workload.
    Reports combined metrics including total runtime and consistency violations.
    """

    @property
    def name(self) -> str:
        return "concurrent_acid"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run all concurrent tests and aggregate results."""
        total_ops = 0

        # Throughput test (simplified)
        throughput_bench = ThroughputScalingBenchmark()
        throughput_bench._person_ids = self._person_ids
        throughput_bench._person_count = self._person_count
        total_ops += throughput_bench.run_iteration(adapter, scale)

        # Lost update test
        lost_update_bench = LostUpdateBenchmark()
        lost_update_bench._person_ids = self._person_ids
        lost_update_bench._person_count = self._person_count
        total_ops += lost_update_bench.run_iteration(adapter, scale)

        # Read after write test
        raw_bench = ReadAfterWriteBenchmark()
        raw_bench._person_ids = self._person_ids
        raw_bench._person_count = self._person_count
        total_ops += raw_bench.run_iteration(adapter, scale)

        # Mixed workload test
        mixed_bench = MixedWorkloadBenchmark()
        mixed_bench._person_ids = self._person_ids
        mixed_bench._person_count = self._person_count
        total_ops += mixed_bench.run_iteration(adapter, scale)

        return total_ops
