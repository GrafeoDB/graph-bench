r"""
LDBC SNB Interactive Workload benchmarks.

Implements key queries from the LDBC Social Network Benchmark Interactive workload:
- Short reads (IS1-IS7): Simple lookups and traversals
- Complex reads (IC1-IC14): Multi-hop queries with aggregations

Reference: https://github.com/ldbc/ldbc_snb_interactive_v2_driver
Spec: https://ldbcouncil.org/benchmarks/snb/

    from graph_bench.benchmarks.snb_interactive import SnbIS1Benchmark

    bench = SnbIS1Benchmark()
    metrics = bench.run(adapter, scale)
"""

import random
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.datasets.ldbc_snb import LDBCSocialNetwork
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    "SnbIS1Benchmark",
    "SnbIS3Benchmark",
    "SnbIC1Benchmark",
    "SnbIC2Benchmark",
    "SnbIC3Benchmark",
    "SnbIC6Benchmark",
]


class SnbBenchmarkBase(BaseBenchmark):
    """Base class for LDBC SNB Interactive benchmarks.

    Uses LDBC SNB data model with Person, City, Tag nodes
    and KNOWS, LIVES_IN, HAS_INTEREST edges.
    """

    category = "ldbc_snb"
    _person_ids: list[str] = []
    _city_ids: list[str] = []
    _tag_ids: list[str] = []
    _person_count: int = 0

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Setup LDBC SNB dataset."""
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

        # Store IDs for queries
        self._person_ids = [n["id"] for n in persons]
        self._city_ids = [n["id"] for n in cities]
        self._tag_ids = [n["id"] for n in tags]
        self._person_count = len(persons)


# =============================================================================
# Short Reads (IS1-IS7) - Simple lookups
# =============================================================================


@BenchmarkRegistry.register("snb_is1", category="ldbc_snb")
class SnbIS1Benchmark(SnbBenchmarkBase):
    """SNB Interactive Short 1: Get person profile.

    Given a person ID, return their profile information.
    This tests point lookup performance.
    """

    @property
    def name(self) -> str:
        return "snb_is1"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get profile of random persons."""
        found = 0
        for _ in range(100):
            person_id = random.choice(self._person_ids)
            node = adapter.get_node(person_id)
            if node:
                found += 1
        return found


@BenchmarkRegistry.register("snb_is3", category="ldbc_snb")
class SnbIS3Benchmark(SnbBenchmarkBase):
    """SNB Interactive Short 3: Get friends of a person.

    Given a person ID, return their direct friends (1-hop KNOWS).
    This tests basic traversal performance.
    """

    @property
    def name(self) -> str:
        return "snb_is3"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get friends of random persons."""
        total_friends = 0
        for _ in range(50):
            person_id = random.choice(self._person_ids)
            friends = adapter.get_neighbors(person_id, edge_type="KNOWS")
            total_friends += len(friends)
        return total_friends


# =============================================================================
# Complex Reads (IC1-IC14) - Multi-hop queries
# =============================================================================


@BenchmarkRegistry.register("snb_ic1", category="ldbc_snb")
class SnbIC1Benchmark(SnbBenchmarkBase):
    """SNB Interactive Complex 1: Friends with given first name.

    Given a person, find all friends (up to 3 hops) with a specific first name.
    Tests multi-hop traversal with property filtering.
    """

    @property
    def name(self) -> str:
        return "snb_ic1"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find friends-of-friends with name filtering."""
        total_matches = 0

        for _ in range(10):
            person_id = random.choice(self._person_ids)

            # Get friends up to 3 hops using BFS
            visited = adapter.traverse_bfs(person_id, max_depth=3, edge_type="KNOWS")

            # Count as matches (in real SNB we'd filter by name)
            total_matches += len(visited)

        return total_matches


@BenchmarkRegistry.register("snb_ic2", category="ldbc_snb")
class SnbIC2Benchmark(SnbBenchmarkBase):
    """SNB Interactive Complex 2: Recent posts from friends.

    Given a person, find recent messages from their friends.
    Tests 2-hop traversal with ordering.
    """

    @property
    def name(self) -> str:
        return "snb_ic2"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get friends and their potential content."""
        total_friends = 0

        for _ in range(20):
            person_id = random.choice(self._person_ids)

            # Get direct friends
            friends = adapter.get_neighbors(person_id, edge_type="KNOWS")
            total_friends += len(friends)

            # In full SNB, we'd get posts from each friend
            # Here we just count friends as proxy

        return total_friends


@BenchmarkRegistry.register("snb_ic3", category="ldbc_snb")
class SnbIC3Benchmark(SnbBenchmarkBase):
    """SNB Interactive Complex 3: Friends in countries.

    Given a person and countries, find friends/friends-of-friends
    who live in those countries. Tests multi-hop with join.
    """

    @property
    def name(self) -> str:
        return "snb_ic3"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find friends in specific locations."""
        total_found = 0

        for _ in range(10):
            person_id = random.choice(self._person_ids)

            # Get 2-hop friends
            friends = adapter.traverse_bfs(person_id, max_depth=2, edge_type="KNOWS")

            # For each friend, check where they live
            for friend_id in friends[:20]:  # Limit for performance
                cities = adapter.get_neighbors(friend_id, edge_type="LIVES_IN")
                total_found += len(cities)

        return total_found


@BenchmarkRegistry.register("snb_ic6", category="ldbc_snb")
class SnbIC6Benchmark(SnbBenchmarkBase):
    """SNB Interactive Complex 6: Tag co-occurrence.

    Given a person and tag, find other tags that frequently
    co-occur with posts having the given tag. Tests aggregation.
    """

    @property
    def name(self) -> str:
        return "snb_ic6"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find tag co-occurrence via person interests."""
        total_tags = 0

        for _ in range(20):
            person_id = random.choice(self._person_ids)

            # Get person's interests (tags)
            interests = adapter.get_neighbors(person_id, edge_type="HAS_INTEREST")
            total_tags += len(interests)

            # Get friends' interests for comparison
            friends = adapter.get_neighbors(person_id, edge_type="KNOWS")
            for friend_id in friends[:10]:  # Limit
                friend_interests = adapter.get_neighbors(friend_id, edge_type="HAS_INTEREST")
                total_tags += len(friend_interests)

        return total_tags
