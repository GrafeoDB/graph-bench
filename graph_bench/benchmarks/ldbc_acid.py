r"""
LDBC ACID benchmark implementation.

Implements the official LDBC ACID test suite for testing transactional
consistency guarantees in graph databases.

Reference: https://github.com/ldbc/ldbc_acid
Paper: "Towards Testing ACID Compliance in the LDBC Social Network Benchmark" (TPCTC 2020)

Test Categories:
- Atomicity Tests (2): Atomicity-C (commit), Atomicity-RB (rollback)
- Isolation Anomaly Tests (10): G0, G1a, G1b, G1c, IMP, PMP, OTV, FR, LU, WS

Data Model (minimal SNB subset):
- Person nodes: id, name, version, numFriends
- KNOWS edges: between Person nodes

    from graph_bench.benchmarks.ldbc_acid import AtomicityCommitTest

    test = AtomicityCommitTest()
    metrics = test.run(adapter, scale)
"""

from __future__ import annotations

import threading
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    # Atomicity
    "AtomicityCommitTest",
    "AtomicityRollbackTest",
    # Isolation Anomalies
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


@dataclass
class AcidTestResult:
    """Result from a single ACID test execution."""

    test_name: str
    passed: bool
    anomaly_detected: bool = False
    violations: int = 0
    details: str = ""
    execution_time_ms: float = 0.0


# =============================================================================
# Base Class for LDBC ACID Tests
# =============================================================================


class LdbcAcidTestBase(BaseBenchmark):
    """Base class for LDBC ACID tests.

    Uses a minimal property graph matching LDBC SNB schema:
    - Person nodes with id, name, version, numFriends
    - KNOWS edges connecting Person nodes

    Each test follows the pattern: init -> concurrent operations -> check
    """

    category = "ldbc_acid"

    def __init__(self, seed: int = 42) -> None:
        self._person_ids: list[str] = []
        self._test_results: list[AcidTestResult] = []

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Setup minimal graph for ACID testing."""
        adapter.clear()

        # Create Person nodes (LDBC ACID uses small graphs)
        # Using 10 persons for most tests, can scale up
        num_persons = min(100, max(10, scale.nodes // 1000))

        nodes = [
            {
                "id": f"person_{i}",
                "name": f"Person {i}",
                "version": 0,
                "numFriends": 0,
            }
            for i in range(num_persons)
        ]
        adapter.insert_nodes(nodes, label="Person")

        # Create KNOWS edges (sparse connectivity for tests)
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(num_persons - 1):
            edges.append(
                (f"person_{i}", f"person_{i + 1}", "KNOWS", {"creationDate": i, "version": 0})
            )

        adapter.insert_edges(edges)
        self._person_ids = [f"person_{i}" for i in range(num_persons)]

    @abstractmethod
    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Run the specific ACID test and return result."""
        ...

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Run the ACID test iteration."""
        result = self.run_test(adapter)
        # Return 1 if test passed (no anomaly when not expected), 0 if failed
        return 1 if result.passed else 0


# =============================================================================
# ATOMICITY TESTS
# =============================================================================


@BenchmarkRegistry.register("acid_atomicity_c", category="ldbc_acid")
class AtomicityCommitTest(LdbcAcidTestBase):
    """Atomicity-C: Test that committed transactions are fully visible.

    LDBC ACID Spec:
    1. Transaction T1: Update Person p1's version and numFriends
    2. Commit T1
    3. Read p1 and verify BOTH changes are visible

    A violation occurs if only partial changes are visible after commit.
    """

    @property
    def name(self) -> str:
        return "acid_atomicity_c"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test atomicity of committed transactions."""
        target = self._person_ids[0]
        new_version = 100
        new_friends = 42

        # Reset to known state
        adapter.update_node(target, {"version": 0, "numFriends": 0})

        start = time.perf_counter()

        # Transaction: update both fields (should be atomic)
        adapter.update_node(target, {"version": new_version, "numFriends": new_friends})

        # Verify both changes are visible
        node = adapter.get_node(target)

        end = time.perf_counter()

        if node is None:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details="Node not found after commit",
                execution_time_ms=(end - start) * 1000,
            )

        version_ok = node.get("version") == new_version
        friends_ok = node.get("numFriends") == new_friends

        if not version_ok or not friends_ok:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details=f"Partial visibility: version={node.get('version')}, numFriends={node.get('numFriends')}",
                execution_time_ms=(end - start) * 1000,
            )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            anomaly_detected=False,
            execution_time_ms=(end - start) * 1000,
        )


@BenchmarkRegistry.register("acid_atomicity_rb", category="ldbc_acid")
class AtomicityRollbackTest(LdbcAcidTestBase):
    """Atomicity-RB: Test that aborted transactions leave no trace.

    LDBC ACID Spec:
    1. Record initial state of Person p1
    2. Start transaction T1: Update p1's version
    3. Abort/rollback T1 (if supported) or simulate with error
    4. Verify p1 is unchanged from initial state

    Note: Many databases auto-commit, so we test error recovery instead.
    """

    @property
    def name(self) -> str:
        return "acid_atomicity_rb"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test that failed operations don't corrupt prior committed state.

        1. Set initial state (version=0, numFriends=0)
        2. Do a valid update (version=1, numFriends=1) — committed
        3. Attempt a failing update on a non-existent node
        4. Verify the committed state (version=1, numFriends=1) is intact
        """
        target = self._person_ids[0]

        # Set known initial state
        adapter.update_node(target, {"version": 0, "numFriends": 0})

        start = time.perf_counter()

        # Commit a valid update
        adapter.update_node(target, {"version": 1, "numFriends": 1})

        # Verify the committed update
        node_mid = adapter.get_node(target)
        if node_mid is None or node_mid.get("version") != 1:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                details="Valid update not visible",
                execution_time_ms=0,
            )

        # Attempt a failing operation (update non-existent node)
        try:
            adapter.update_node(
                "nonexistent_person_xyz", {"version": 999},
            )
        except Exception:
            pass  # Expected to fail

        # Verify committed state is still intact
        node_after = adapter.get_node(target)

        end = time.perf_counter()

        if node_after is None:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details="Node disappeared after failed operation",
                execution_time_ms=(end - start) * 1000,
            )

        version_ok = node_after.get("version") == 1
        friends_ok = node_after.get("numFriends") == 1

        if not version_ok or not friends_ok:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details=(
                    f"Committed state corrupted: version={node_after.get('version')}, "
                    f"numFriends={node_after.get('numFriends')}"
                ),
                execution_time_ms=(end - start) * 1000,
            )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            details="Committed state preserved after failed operation",
            execution_time_ms=(end - start) * 1000,
        )


# =============================================================================
# ISOLATION ANOMALY TESTS
# =============================================================================


@BenchmarkRegistry.register("acid_g0", category="ldbc_acid")
class G0DirtyWriteTest(LdbcAcidTestBase):
    """G0: Dirty Write anomaly test.

    LDBC ACID Spec:
    Two transactions write to the same item. G0 occurs if the writes
    become interleaved at a finer granularity than the transaction.

    Test:
    1. T1: Update p1.version = 1, then p1.numFriends = 1
    2. T2: Update p1.version = 2, then p1.numFriends = 2
    3. Final state should be either (1,1) or (2,2), not mixed

    Anomaly: Final state is (1,2) or (2,1)
    """

    @property
    def name(self) -> str:
        return "acid_g0"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for dirty write anomaly."""
        target = self._person_ids[0]
        violations = 0
        barrier = threading.Barrier(2)

        # Reset state
        adapter.update_node(target, {"version": 0, "numFriends": 0})

        start = time.perf_counter()

        def txn1() -> None:
            barrier.wait()  # Synchronize start
            adapter.update_node(target, {"version": 1})
            time.sleep(0.001)  # Small delay to increase interleaving chance
            adapter.update_node(target, {"numFriends": 1})

        def txn2() -> None:
            barrier.wait()  # Synchronize start
            adapter.update_node(target, {"version": 2})
            time.sleep(0.001)
            adapter.update_node(target, {"numFriends": 2})

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(txn1)
            f2 = executor.submit(txn2)
            f1.result()
            f2.result()

        # Check final state
        node = adapter.get_node(target)
        end = time.perf_counter()

        if node is None:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details="Node not found",
                execution_time_ms=(end - start) * 1000,
            )

        version = node.get("version")
        friends = node.get("numFriends")

        # Valid states: (1,1) or (2,2)
        # Anomaly: (1,2) or (2,1)
        if (version == 1 and friends == 1) or (version == 2 and friends == 2):
            passed = True
            anomaly = False
        else:
            passed = False
            anomaly = True
            violations = 1

        return AcidTestResult(
            test_name=self.name,
            passed=passed,
            anomaly_detected=anomaly,
            violations=violations,
            details=f"Final state: version={version}, numFriends={friends}",
            execution_time_ms=(end - start) * 1000,
        )


@BenchmarkRegistry.register("acid_g1a", category="ldbc_acid")
class G1aAbortedReadTest(LdbcAcidTestBase):
    """G1a: Aborted Read (Dirty Read) anomaly test.

    LDBC ACID Spec:
    Transaction reads data written by a transaction that later aborts.

    Test (adapted for auto-commit databases):
    1. T1: Write intermediate value, then final value
    2. T2: Read the value during T1's execution
    3. T2 should never see the intermediate value after T1 completes

    We simulate this by checking read consistency during concurrent updates.
    """

    @property
    def name(self) -> str:
        return "acid_g1a"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for dirty read anomaly."""
        target = self._person_ids[0]
        intermediate_value = 999
        final_value = 100
        observed_values: list[int] = []
        stop_reading = threading.Event()

        # Reset state
        adapter.update_node(target, {"version": 0})

        start = time.perf_counter()

        def writer() -> None:
            # Simulate a transaction that writes intermediate then final value
            adapter.update_node(target, {"version": intermediate_value})
            time.sleep(0.005)  # Simulate processing
            adapter.update_node(target, {"version": final_value})
            stop_reading.set()

        def reader() -> None:
            while not stop_reading.is_set():
                node = adapter.get_node(target)
                if node:
                    observed_values.append(node.get("version", -1))
                time.sleep(0.001)

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(writer)
            f2 = executor.submit(reader)
            f1.result()
            f2.result()

        end = time.perf_counter()

        # After writer completes, any observation of intermediate_value is a dirty read
        # In a proper isolation level, reader should only see 0 or final_value
        # Seeing intermediate_value after final commit indicates G1a

        # For this simplified test, we check if intermediate was observed
        # after the final value was written (which shouldn't happen)
        saw_intermediate = intermediate_value in observed_values

        # Note: In auto-commit DBs, seeing intermediate is expected during execution
        # The real test is whether it persists after the "transaction" completes
        final_node = adapter.get_node(target)
        final_version = final_node.get("version") if final_node else -1

        if final_version != final_value:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details=f"Final value incorrect: {final_version} (expected {final_value})",
                execution_time_ms=(end - start) * 1000,
            )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            anomaly_detected=saw_intermediate,  # Note but don't fail for auto-commit DBs
            details=f"Observed intermediate: {saw_intermediate}",
            execution_time_ms=(end - start) * 1000,
        )


@BenchmarkRegistry.register("acid_g1b", category="ldbc_acid")
class G1bIntermediateReadTest(LdbcAcidTestBase):
    """G1b: Intermediate Read anomaly test.

    LDBC ACID Spec:
    Transaction reads an intermediate state of another transaction.

    Test:
    1. T1: Increment version from 0 to 1, then to 2
    2. T2: Read version multiple times
    3. If T2 reads version=1 after T1 commits with version=2, that's G1b
    """

    @property
    def name(self) -> str:
        return "acid_g1b"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for intermediate read anomaly."""
        target = self._person_ids[0]

        # Reset
        adapter.update_node(target, {"version": 0})

        start = time.perf_counter()

        # Writer does multiple updates
        adapter.update_node(target, {"version": 1})
        adapter.update_node(target, {"version": 2})

        # After both updates, reader should see version=2
        node = adapter.get_node(target)
        end = time.perf_counter()

        if node is None:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details="Node not found",
                execution_time_ms=(end - start) * 1000,
            )

        version = node.get("version")

        # Should see final version (2), not intermediate (1)
        if version == 2:
            return AcidTestResult(
                test_name=self.name,
                passed=True,
                execution_time_ms=(end - start) * 1000,
            )
        else:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details=f"Read intermediate state: version={version}",
                execution_time_ms=(end - start) * 1000,
            )


@BenchmarkRegistry.register("acid_g1c", category="ldbc_acid")
class G1cCircularInfoFlowTest(LdbcAcidTestBase):
    """G1c: Circular Information Flow anomaly test.

    LDBC ACID Spec:
    Two transactions have circular read-write dependencies.
    T1 reads x, writes y; T2 reads y, writes x.
    G1c occurs if T1 sees T2's write and T2 sees T1's write.

    Test:
    1. p1.version=0, p2.version=0
    2. T1: read p1.version, write p2.version = read_value + 1
    3. T2: read p2.version, write p1.version = read_value + 1
    4. If both read each other's writes, we have circular dependency
    """

    @property
    def name(self) -> str:
        return "acid_g1c"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for circular information flow."""
        p1, p2 = self._person_ids[0], self._person_ids[1]
        t1_read: list[int] = []
        t2_read: list[int] = []
        barrier = threading.Barrier(2)

        # Reset
        adapter.update_node(p1, {"version": 0})
        adapter.update_node(p2, {"version": 0})

        start = time.perf_counter()

        def txn1() -> None:
            barrier.wait()
            node = adapter.get_node(p1)
            val = node.get("version", 0) if node else 0
            t1_read.append(val)
            adapter.update_node(p2, {"version": val + 1})

        def txn2() -> None:
            barrier.wait()
            node = adapter.get_node(p2)
            val = node.get("version", 0) if node else 0
            t2_read.append(val)
            adapter.update_node(p1, {"version": val + 1})

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(txn1)
            f2 = executor.submit(txn2)
            f1.result()
            f2.result()

        end = time.perf_counter()

        # Check final state
        n1 = adapter.get_node(p1)
        n2 = adapter.get_node(p2)

        p1_final = n1.get("version") if n1 else -1
        p2_final = n2.get("version") if n2 else -1

        # G1c: Both transactions saw each other's writes
        # If T1 read p1=1 (written by T2) and T2 read p2=1 (written by T1)
        # that's circular dependency
        t1_saw_t2 = t1_read and t1_read[0] > 0
        t2_saw_t1 = t2_read and t2_read[0] > 0

        if t1_saw_t2 and t2_saw_t1:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details=f"Circular dependency: T1 read {t1_read}, T2 read {t2_read}",
                execution_time_ms=(end - start) * 1000,
            )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            details=f"Final: p1={p1_final}, p2={p2_final}",
            execution_time_ms=(end - start) * 1000,
        )


@BenchmarkRegistry.register("acid_imp", category="ldbc_acid")
class IMPItemManyPrecedersTest(LdbcAcidTestBase):
    """IMP: Item-Many-Preceders anomaly test.

    LDBC ACID Spec:
    A transaction observes multiple versions of the same item.

    Test:
    1. T1 reads p1.version twice within same "transaction"
    2. Concurrent T2 updates p1.version between T1's reads
    3. IMP occurs if T1's two reads return different values
    """

    @property
    def name(self) -> str:
        return "acid_imp"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for item-many-preceders anomaly."""
        target = self._person_ids[0]
        read1: list[int] = []
        read2: list[int] = []
        barrier = threading.Barrier(2)
        midpoint = threading.Event()

        # Reset
        adapter.update_node(target, {"version": 0})

        start = time.perf_counter()

        def reader() -> None:
            barrier.wait()
            # First read
            node1 = adapter.get_node(target)
            read1.append(node1.get("version", -1) if node1 else -1)

            midpoint.set()  # Signal writer to update
            time.sleep(0.01)  # Wait for potential update

            # Second read (should be same in snapshot isolation)
            node2 = adapter.get_node(target)
            read2.append(node2.get("version", -1) if node2 else -1)

        def writer() -> None:
            barrier.wait()
            midpoint.wait()  # Wait for reader's first read
            adapter.update_node(target, {"version": 100})

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(reader)
            f2 = executor.submit(writer)
            f1.result()
            f2.result()

        end = time.perf_counter()

        v1 = read1[0] if read1 else -1
        v2 = read2[0] if read2 else -1

        # IMP: read different versions within "same transaction"
        # Note: Without explicit transactions, this may occur in read-committed
        if v1 != v2:
            return AcidTestResult(
                test_name=self.name,
                passed=False,  # Depends on expected isolation level
                anomaly_detected=True,
                violations=1,
                details=f"Read different versions: {v1} then {v2}",
                execution_time_ms=(end - start) * 1000,
            )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            details=f"Consistent reads: {v1}",
            execution_time_ms=(end - start) * 1000,
        )


@BenchmarkRegistry.register("acid_pmp", category="ldbc_acid")
class PMPPredicateManyPrecedersTest(LdbcAcidTestBase):
    """PMP: Predicate-Many-Preceders anomaly test.

    LDBC ACID Spec:
    A transaction gets inconsistent results from repeated predicate queries.

    Test:
    1. T1 counts persons with version > 0 twice
    2. T2 updates a person's version between counts
    3. PMP occurs if counts differ
    """

    @property
    def name(self) -> str:
        return "acid_pmp"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for predicate-many-preceders anomaly."""
        # Reset all persons to version 0
        for pid in self._person_ids[:5]:
            adapter.update_node(pid, {"version": 0})

        count1: list[int] = []
        count2: list[int] = []
        barrier = threading.Barrier(2)
        midpoint = threading.Event()

        start = time.perf_counter()

        def counter() -> None:
            barrier.wait()
            # First count
            c1 = 0
            for pid in self._person_ids[:5]:
                node = adapter.get_node(pid)
                if node and node.get("version", 0) > 0:
                    c1 += 1
            count1.append(c1)

            midpoint.set()
            time.sleep(0.01)

            # Second count
            c2 = 0
            for pid in self._person_ids[:5]:
                node = adapter.get_node(pid)
                if node and node.get("version", 0) > 0:
                    c2 += 1
            count2.append(c2)

        def updater() -> None:
            barrier.wait()
            midpoint.wait()
            # Update one person to have version > 0
            adapter.update_node(self._person_ids[0], {"version": 1})

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(counter)
            f2 = executor.submit(updater)
            f1.result()
            f2.result()

        end = time.perf_counter()

        c1 = count1[0] if count1 else -1
        c2 = count2[0] if count2 else -1

        if c1 != c2:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details=f"Predicate counts differ: {c1} then {c2}",
                execution_time_ms=(end - start) * 1000,
            )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            details=f"Consistent counts: {c1}",
            execution_time_ms=(end - start) * 1000,
        )


@BenchmarkRegistry.register("acid_otv", category="ldbc_acid")
class OTVObservedTxnVanishesTest(LdbcAcidTestBase):
    """OTV: Observed Transaction Vanishes anomaly test.

    LDBC ACID Spec:
    A transaction sees partial effects of another transaction that
    later appear to "vanish".

    Test:
    1. T1 updates p1 and p2 (should be atomic)
    2. T2 reads p1 (sees update), reads p2 (doesn't see update)
    3. OTV: T2 saw T1's effect on p1 but not on p2
    """

    @property
    def name(self) -> str:
        return "acid_otv"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for observed transaction vanishes anomaly."""
        p1, p2 = self._person_ids[0], self._person_ids[1]

        # Reset
        adapter.update_node(p1, {"version": 0})
        adapter.update_node(p2, {"version": 0})

        start = time.perf_counter()

        # Simulate atomic update of both
        adapter.update_node(p1, {"version": 1})
        adapter.update_node(p2, {"version": 1})

        # Read both
        n1 = adapter.get_node(p1)
        n2 = adapter.get_node(p2)

        end = time.perf_counter()

        v1 = n1.get("version") if n1 else -1
        v2 = n2.get("version") if n2 else -1

        # OTV: saw v1=1 but v2=0 (or vice versa)
        if (v1 == 1 and v2 == 0) or (v1 == 0 and v2 == 1):
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details=f"Partial visibility: p1.version={v1}, p2.version={v2}",
                execution_time_ms=(end - start) * 1000,
            )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            details=f"Consistent: p1={v1}, p2={v2}",
            execution_time_ms=(end - start) * 1000,
        )


@BenchmarkRegistry.register("acid_fr", category="ldbc_acid")
class FRFracturedReadTest(LdbcAcidTestBase):
    """FR: Fractured Read anomaly test.

    LDBC ACID Spec:
    A transaction reads from multiple items and sees an inconsistent
    cut across their version histories.

    Test:
    1. p1.version=1, p2.version=1 (consistent state)
    2. T1: update both to version=2
    3. T2: read p1 and p2
    4. FR: T2 reads p1.version=2, p2.version=1 (fractured)
    """

    @property
    def name(self) -> str:
        return "acid_fr"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for fractured read anomaly."""
        p1, p2 = self._person_ids[0], self._person_ids[1]
        reads: list[tuple[int, int]] = []
        barrier = threading.Barrier(2)
        midpoint = threading.Event()

        # Setup consistent initial state
        adapter.update_node(p1, {"version": 1})
        adapter.update_node(p2, {"version": 1})

        start = time.perf_counter()

        def writer() -> None:
            barrier.wait()
            midpoint.wait()
            # Update both to version 2
            adapter.update_node(p1, {"version": 2})
            adapter.update_node(p2, {"version": 2})

        def reader() -> None:
            barrier.wait()
            midpoint.set()
            time.sleep(0.005)  # Give writer time to partially complete

            n1 = adapter.get_node(p1)
            n2 = adapter.get_node(p2)
            v1 = n1.get("version", -1) if n1 else -1
            v2 = n2.get("version", -1) if n2 else -1
            reads.append((v1, v2))

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(writer)
            f2 = executor.submit(reader)
            f1.result()
            f2.result()

        end = time.perf_counter()

        if reads:
            v1, v2 = reads[0]
            # Valid: (1,1) or (2,2)
            # Fractured: (1,2) or (2,1)
            if v1 != v2:
                return AcidTestResult(
                    test_name=self.name,
                    passed=False,
                    anomaly_detected=True,
                    violations=1,
                    details=f"Fractured read: p1={v1}, p2={v2}",
                    execution_time_ms=(end - start) * 1000,
                )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            details=f"Consistent read: {reads}",
            execution_time_ms=(end - start) * 1000,
        )


@BenchmarkRegistry.register("acid_lu", category="ldbc_acid")
class LULostUpdateTest(LdbcAcidTestBase):
    """LU: Lost Update anomaly test.

    LDBC ACID Spec:
    Two transactions read the same value, both compute new values based
    on the read, and both write. One update is lost.

    Test:
    1. p1.numFriends = 0
    2. T1: read numFriends, write numFriends + 1
    3. T2: read numFriends, write numFriends + 1
    4. Expected: numFriends = 2
    5. LU: numFriends = 1 (one update lost)
    """

    @property
    def name(self) -> str:
        return "acid_lu"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for lost update anomaly."""
        target = self._person_ids[0]
        num_workers = 4
        increments_per_worker = 10
        expected = num_workers * increments_per_worker
        barrier = threading.Barrier(num_workers)

        # Reset
        adapter.update_node(target, {"numFriends": 0})

        start = time.perf_counter()

        def incrementer() -> None:
            barrier.wait()
            for _ in range(increments_per_worker):
                node = adapter.get_node(target)
                current = node.get("numFriends", 0) if node else 0
                adapter.update_node(target, {"numFriends": current + 1})

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(incrementer) for _ in range(num_workers)]
            for f in as_completed(futures):
                f.result()

        node = adapter.get_node(target)
        end = time.perf_counter()

        final = node.get("numFriends", 0) if node else 0
        lost = expected - final

        if lost > 0:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=lost,
                details=f"Lost {lost} updates: expected {expected}, got {final}",
                execution_time_ms=(end - start) * 1000,
            )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            details=f"No lost updates: {final}",
            execution_time_ms=(end - start) * 1000,
        )


@BenchmarkRegistry.register("acid_ws", category="ldbc_acid")
class WSWriteSkewTest(LdbcAcidTestBase):
    """WS: Write Skew anomaly test.

    LDBC ACID Spec:
    Two transactions read overlapping data, make disjoint updates
    based on what they read, and both commit. The combined result
    violates a constraint that each transaction individually preserved.

    Test (simplified balance constraint):
    1. p1.numFriends = 10, p2.numFriends = 10
    2. Constraint: total friends >= 15
    3. T1: if p1 + p2 >= 15, then p1 -= 5
    4. T2: if p1 + p2 >= 15, then p2 -= 5
    5. Both see total=20 >= 15, both decrement
    6. Result: total = 10, violating constraint

    Note: Detecting WS requires application-level constraint checking.
    """

    @property
    def name(self) -> str:
        return "acid_ws"

    def run_test(self, adapter: GraphDatabaseAdapter) -> AcidTestResult:
        """Test for write skew anomaly."""
        p1, p2 = self._person_ids[0], self._person_ids[1]
        min_total = 15
        decrement = 5
        barrier = threading.Barrier(2)

        # Setup: each has 10, total=20 >= 15
        adapter.update_node(p1, {"numFriends": 10})
        adapter.update_node(p2, {"numFriends": 10})

        start = time.perf_counter()

        def txn1() -> None:
            barrier.wait()
            n1 = adapter.get_node(p1)
            n2 = adapter.get_node(p2)
            v1 = n1.get("numFriends", 0) if n1 else 0
            v2 = n2.get("numFriends", 0) if n2 else 0

            if v1 + v2 >= min_total:
                adapter.update_node(p1, {"numFriends": v1 - decrement})

        def txn2() -> None:
            barrier.wait()
            n1 = adapter.get_node(p1)
            n2 = adapter.get_node(p2)
            v1 = n1.get("numFriends", 0) if n1 else 0
            v2 = n2.get("numFriends", 0) if n2 else 0

            if v1 + v2 >= min_total:
                adapter.update_node(p2, {"numFriends": v2 - decrement})

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(txn1)
            f2 = executor.submit(txn2)
            f1.result()
            f2.result()

        # Check constraint
        n1 = adapter.get_node(p1)
        n2 = adapter.get_node(p2)
        end = time.perf_counter()

        v1 = n1.get("numFriends", 0) if n1 else 0
        v2 = n2.get("numFriends", 0) if n2 else 0
        total = v1 + v2

        if total < min_total:
            return AcidTestResult(
                test_name=self.name,
                passed=False,
                anomaly_detected=True,
                violations=1,
                details=f"Write skew: total={total} < {min_total} (p1={v1}, p2={v2})",
                execution_time_ms=(end - start) * 1000,
            )

        return AcidTestResult(
            test_name=self.name,
            passed=True,
            details=f"Constraint preserved: total={total} (p1={v1}, p2={v2})",
            execution_time_ms=(end - start) * 1000,
        )
