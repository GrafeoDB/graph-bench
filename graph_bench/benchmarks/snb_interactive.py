r"""
LDBC SNB Interactive Workload benchmarks.

Implements key queries from the LDBC Social Network Benchmark Interactive workload:
- Short reads (IS1-IS7): Simple lookups and traversals
- Complex reads (IC1-IC14): Multi-hop queries with aggregations

Reference: https://github.com/ldbc/ldbc_snb_interactive_v1_impls
Spec: https://ldbcouncil.org/ldbc_snb_docs/ldbc-snb-specification.pdf

Data Model:
- Person: firstName, lastName, gender, birthday, creationDate, locationIP, browserUsed
- City: name, country
- Tag: name
- Relationships: KNOWS, LIVES_IN, HAS_INTEREST

    from graph_bench.benchmarks.snb_interactive import SnbIS1Benchmark

    bench = SnbIS1Benchmark()
    metrics = bench.run(adapter, scale)
"""

from __future__ import annotations

import random
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.datasets.ldbc_snb import LDBCSocialNetwork
from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = [
    # Short Reads
    "SnbIS1Benchmark",
    "SnbIS2Benchmark",
    "SnbIS3Benchmark",
    "SnbIS4Benchmark",
    "SnbIS5Benchmark",
    "SnbIS6Benchmark",
    "SnbIS7Benchmark",
    # Complex Reads
    "SnbIC1Benchmark",
    "SnbIC2Benchmark",
    "SnbIC3Benchmark",
    "SnbIC6Benchmark",
]


class SnbInteractiveBenchmarkBase(BaseBenchmark):
    """Base class for LDBC SNB Interactive benchmarks.

    Uses LDBC SNB data model with Person, City, Tag nodes
    and KNOWS, LIVES_IN, HAS_INTEREST edges.

    Instance variables used to avoid class-level state issues.
    """

    category = "ldbc_snb"

    def __init__(self, seed: int = 42) -> None:
        """Initialize benchmark with reproducible random seed."""
        self._seed = seed
        self._person_ids: list[str] = []
        self._city_ids: list[str] = []
        self._tag_ids: list[str] = []
        self._person_count: int = 0
        self._rng = random.Random(seed)

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Setup LDBC SNB dataset.

        Creates a social network graph with:
        - Person nodes with demographic properties
        - City nodes with location info
        - Tag nodes for interests
        - KNOWS, LIVES_IN, HAS_INTEREST relationships
        """
        adapter.clear()

        # Reset RNG for reproducibility
        self._rng = random.Random(self._seed)

        # Generate SNB data using the dataset generator
        dataset = LDBCSocialNetwork(scale_factor=1, seed=self._seed)
        nodes, edges = dataset.generate(scale)

        # Separate nodes by label
        persons = [n for n in nodes if n.get("label") == "Person"]
        cities = [n for n in nodes if n.get("label") == "City"]
        tags = [n for n in nodes if n.get("label") == "Tag"]

        # Insert nodes by label
        for label, node_list in [("Person", persons), ("City", cities), ("Tag", tags)]:
            if node_list:
                adapter.insert_nodes(node_list, label=label)

        # Insert edges
        adapter.insert_edges(edges)

        # Store IDs for queries (using instance variables)
        self._person_ids = [n["id"] for n in persons]
        self._city_ids = [n["id"] for n in cities]
        self._tag_ids = [n["id"] for n in tags]
        self._person_count = len(persons)

    def _get_person(self, index: int = 0) -> str:
        """Get a deterministic person ID for queries.

        Uses index to allow multiple queries with different starting points
        while maintaining reproducibility.
        """
        if not self._person_ids:
            return "person_0"
        return self._person_ids[index % len(self._person_ids)]

    def _get_city(self, index: int = 0) -> str:
        """Get a deterministic city ID."""
        if not self._city_ids:
            return "city_0"
        return self._city_ids[index % len(self._city_ids)]

    def _get_tag(self, index: int = 0) -> str:
        """Get a deterministic tag ID."""
        if not self._tag_ids:
            return "tag_0"
        return self._tag_ids[index % len(self._tag_ids)]


# =============================================================================
# Short Reads (IS1-IS7) - Simple lookups
# =============================================================================


@BenchmarkRegistry.register("snb_is1", category="ldbc_snb")
class SnbIS1Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Short 1: Profile of a Person.

    LDBC Spec:
    Given a Person's ID, retrieve their profile information:
    firstName, lastName, birthday, locationIP, browserUsed, cityId, gender, creationDate

    Tests: Point lookup performance.
    """

    @property
    def name(self) -> str:
        return "snb_is1"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get profile of persons using deterministic selection."""
        found = 0
        # Run 100 lookups with deterministic person selection
        for i in range(100):
            person_id = self._get_person(i)
            node = adapter.get_node(person_id)
            if node:
                found += 1
        return found


@BenchmarkRegistry.register("snb_is2", category="ldbc_snb")
class SnbIS2Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Short 2: Recent Messages of a Person.

    LDBC Spec:
    Given a Person's ID, retrieve 10 most recent Messages created by that Person.
    Return messageId, messageContent/imageFile, creationDate, originalPostId,
    originalPostAuthorId, originalPostAuthorFirstName, originalPostAuthorLastName.

    Simplified: We retrieve the person's recent activity (neighbors).
    Tests: Single-hop traversal with ordering.
    """

    @property
    def name(self) -> str:
        return "snb_is2"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get recent activity for persons."""
        total_activity = 0
        for i in range(50):
            person_id = self._get_person(i)
            # Get all outgoing relationships (simulating message lookup)
            neighbors = adapter.get_neighbors(person_id)
            total_activity += len(neighbors)
        return total_activity


@BenchmarkRegistry.register("snb_is3", category="ldbc_snb")
class SnbIS3Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Short 3: Friends of a Person.

    LDBC Spec:
    Given a Person's ID, retrieve all their friends (KNOWS relationships).
    Return personId, firstName, lastName, friendshipCreationDate.
    Order by creationDate DESC, personId ASC.

    Tests: Single-hop traversal on KNOWS edges.
    """

    @property
    def name(self) -> str:
        return "snb_is3"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get friends of persons."""
        total_friends = 0
        for i in range(50):
            person_id = self._get_person(i)
            friends = adapter.get_neighbors(person_id, edge_type="KNOWS")
            total_friends += len(friends)
        return total_friends


@BenchmarkRegistry.register("snb_is4", category="ldbc_snb")
class SnbIS4Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Short 4: Content of a Message.

    LDBC Spec:
    Given a Message ID, retrieve its content and creation date.
    Return messageContent/imageFile, creationDate.

    Simplified: Point lookup of a node.
    Tests: Simple node retrieval.
    """

    @property
    def name(self) -> str:
        return "snb_is4"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get content (properties) of persons as proxy for messages."""
        found = 0
        for i in range(100):
            person_id = self._get_person(i)
            node = adapter.get_node(person_id)
            if node and node.get("firstName"):  # Check property exists
                found += 1
        return found


@BenchmarkRegistry.register("snb_is5", category="ldbc_snb")
class SnbIS5Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Short 5: Creator of a Message.

    LDBC Spec:
    Given a Message ID, retrieve the Person who created it.
    Return personId, firstName, lastName.

    Simplified: Find a node and its creator relationship.
    Tests: Reverse edge traversal.
    """

    @property
    def name(self) -> str:
        return "snb_is5"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get creator relationship (simulated via KNOWS reverse lookup)."""
        found = 0
        for i in range(50):
            person_id = self._get_person(i)
            # Get any neighbor and check we can retrieve their info
            neighbors = adapter.get_neighbors(person_id, edge_type="KNOWS")
            for neighbor_id in neighbors[:5]:
                neighbor = adapter.get_node(neighbor_id)
                if neighbor:
                    found += 1
        return found


@BenchmarkRegistry.register("snb_is6", category="ldbc_snb")
class SnbIS6Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Short 6: Forum of a Message.

    LDBC Spec:
    Given a Message ID, find the Forum it belongs to and the Forum's moderator.
    Return forumId, forumTitle, moderatorId, moderatorFirstName, moderatorLastName.

    Simplified: Multi-hop lookup via LIVES_IN (person -> city).
    Tests: Property lookup with join.
    """

    @property
    def name(self) -> str:
        return "snb_is6"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get location context for persons."""
        found = 0
        for i in range(50):
            person_id = self._get_person(i)
            # Get where person lives
            cities = adapter.get_neighbors(person_id, edge_type="LIVES_IN")
            for city_id in cities:
                city = adapter.get_node(city_id)
                if city and city.get("name"):
                    found += 1
        return found


@BenchmarkRegistry.register("snb_is7", category="ldbc_snb")
class SnbIS7Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Short 7: Replies of a Message.

    LDBC Spec:
    Given a Message ID, find all Comments that are replies to it.
    Return commentId, commentContent, creationDate, replyAuthorId,
    replyAuthorFirstName, replyAuthorLastName, isKnows (whether author knows original poster).

    Simplified: Get neighbors and their properties.
    Tests: Single-hop with property retrieval and join check.
    """

    @property
    def name(self) -> str:
        return "snb_is7"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get replies/comments simulation via friends with details."""
        total_replies = 0
        for i in range(30):
            person_id = self._get_person(i)
            # Get friends
            friends = adapter.get_neighbors(person_id, edge_type="KNOWS")
            for friend_id in friends[:10]:
                # Check if friend knows the person (mutual)
                friend_friends = adapter.get_neighbors(friend_id, edge_type="KNOWS")
                if person_id in friend_friends:
                    total_replies += 1
        return total_replies


# =============================================================================
# Complex Reads (IC1-IC14) - Multi-hop queries
# =============================================================================


@BenchmarkRegistry.register("snb_ic1", category="ldbc_snb")
class SnbIC1Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Complex 1: Transitive Friends with Certain Name.

    LDBC Spec:
    Given a Person's ID and a first name, find all Persons connected within
    3 hops via KNOWS with the given firstName.
    Return personId, lastName, distanceFromPerson, birthday, creationDate,
    gender, browserUsed, locationIP, emails, languages, cityName, universities, companies.

    Tests: Multi-hop traversal with property filtering.
    """

    @property
    def name(self) -> str:
        return "snb_ic1"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find friends-of-friends up to 3 hops."""
        total_matches = 0

        for i in range(10):
            person_id = self._get_person(i)

            # Get friends up to 3 hops using BFS
            visited = adapter.traverse_bfs(person_id, max_depth=3, edge_type="KNOWS")

            # In full LDBC we'd filter by firstName here
            total_matches += len(visited)

        return total_matches


@BenchmarkRegistry.register("snb_ic2", category="ldbc_snb")
class SnbIC2Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Complex 2: Recent Messages by Your Friends.

    LDBC Spec:
    Given a Person and a date, find 20 most recent Messages from their friends
    created before that date.
    Return personId, personFirstName, personLastName, messageId, messageContent,
    messageCreationDate.

    Tests: 2-hop traversal (person->friend->messages) with ordering.
    """

    @property
    def name(self) -> str:
        return "snb_ic2"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Get friends and count their activity."""
        total_activity = 0

        for i in range(20):
            person_id = self._get_person(i)

            # Get direct friends
            friends = adapter.get_neighbors(person_id, edge_type="KNOWS")
            total_activity += len(friends)

            # For each friend, get their outgoing edges (simulating messages)
            for friend_id in friends[:10]:  # Limit for performance
                friend_activity = adapter.get_neighbors(friend_id)
                total_activity += len(friend_activity)

        return total_activity


@BenchmarkRegistry.register("snb_ic3", category="ldbc_snb")
class SnbIC3Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Complex 3: Friends and Friends-of-Friends in Countries.

    LDBC Spec:
    Given a Person and two country names, find friends and friends-of-friends
    from those countries who have interacted within a date range.
    Return personId, personFirstName, personLastName, messageCountX, messageCountY.

    Tests: Multi-hop traversal with location join.
    """

    @property
    def name(self) -> str:
        return "snb_ic3"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find friends in specific locations."""
        total_found = 0

        for i in range(10):
            person_id = self._get_person(i)

            # Get 2-hop friends
            friends_1hop = adapter.get_neighbors(person_id, edge_type="KNOWS")

            for friend_id in friends_1hop[:20]:  # Limit for performance
                # Check where friend lives
                cities = adapter.get_neighbors(friend_id, edge_type="LIVES_IN")
                total_found += len(cities)

                # Get friends-of-friends
                friends_2hop = adapter.get_neighbors(friend_id, edge_type="KNOWS")
                for fof_id in friends_2hop[:5]:
                    fof_cities = adapter.get_neighbors(fof_id, edge_type="LIVES_IN")
                    total_found += len(fof_cities)

        return total_found


@BenchmarkRegistry.register("snb_ic6", category="ldbc_snb")
class SnbIC6Benchmark(SnbInteractiveBenchmarkBase):
    """SNB Interactive Complex 6: Tag Co-occurrence.

    LDBC Spec:
    Given a Person and a Tag, find other Tags that frequently co-occur
    with the given Tag on Messages created by friends and friends-of-friends.
    Return tagName, postCount.

    Tests: Multi-hop traversal with aggregation.
    """

    @property
    def name(self) -> str:
        return "snb_ic6"

    def run_iteration(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> int:
        """Find tag co-occurrence via person interests."""
        total_tags = 0

        for i in range(20):
            person_id = self._get_person(i)

            # Get person's interests (tags)
            interests = adapter.get_neighbors(person_id, edge_type="HAS_INTEREST")
            total_tags += len(interests)

            # Get friends' interests for comparison
            friends = adapter.get_neighbors(person_id, edge_type="KNOWS")
            for friend_id in friends[:10]:  # Limit
                friend_interests = adapter.get_neighbors(friend_id, edge_type="HAS_INTEREST")
                total_tags += len(friend_interests)

        return total_tags
