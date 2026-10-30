r"""
LDBC SNB Interactive Workload benchmarks.

Implements key queries from the LDBC Social Network Benchmark Interactive workload:
- Short reads (IS1-IS7): Simple lookups and traversals
- Complex reads (IC1, IC2, IC3, IC6): Multi-hop queries with aggregations

Reference: https://github.com/ldbc/ldbc_snb_interactive_v1_impls
Spec: https://ldbcouncil.org/ldbc_snb_docs/ldbc-snb-specification.pdf

Data Model (full LDBC SNB schema):
- Person: firstName, lastName, gender, birthday, creationDate, locationIP, browserUsed
- Post: content, imageFile, creationDate, length
- Comment: content, creationDate, length
- Forum: title, creationDate
- City: name, country
- Country: name
- Tag: name
- University: name
- Company: name
- Relationships: KNOWS, LIVES_IN, HAS_INTEREST, HAS_CREATOR, CREATED, REPLY_OF,
  HAS_REPLY, HAS_TAG, CONTAINER_OF, IN_FORUM, HAS_MODERATOR, HAS_MEMBER,
  STUDY_AT, WORK_AT, IS_PART_OF, IS_LOCATED_IN

    from graph_bench.benchmarks.snb_interactive import SnbIS1Benchmark

    bench = SnbIS1Benchmark()
    metrics = bench.run(adapter, scale)
"""

from __future__ import annotations

import random
from typing import Any

from graph_bench.benchmarks.base import BaseBenchmark, BenchmarkRegistry
from graph_bench.datasets.ldbc_snb import (
    FIRST_NAMES_MALE,
    LDBCSocialNetwork,
    scale_name_to_factor,
)
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

# Node labels in the full LDBC SNB schema
_ALL_LABELS = [
    "Person", "City", "Tag", "Country", "University",
    "Company", "Forum", "Post", "Comment",
]


class SnbInteractiveBenchmarkBase(BaseBenchmark):
    """Base class for LDBC SNB Interactive benchmarks.

    Uses the full LDBC SNB data model with Person, Post, Comment, Forum,
    City, Country, Tag, University, Company nodes and all spec relationships.
    """

    category = "ldbc_snb"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._person_ids: list[str] = []
        self._city_ids: list[str] = []
        self._tag_ids: list[str] = []
        self._country_ids: list[str] = []
        self._post_ids: list[str] = []
        self._comment_ids: list[str] = []
        self._message_ids: list[str] = []  # posts + comments interleaved
        self._forum_ids: list[str] = []
        self._person_count: int = 0
        self._rng = random.Random(seed)

    def setup(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig) -> None:
        """Load full LDBC SNB dataset into adapter."""
        adapter.clear()
        self._rng = random.Random(self._seed)

        dataset = LDBCSocialNetwork(
            scale_factor=scale_name_to_factor(scale.name), seed=self._seed,
        )
        nodes, edges = dataset.generate(scale)

        # Group nodes by label and insert
        by_label: dict[str, list[dict[str, Any]]] = {}
        for n in nodes:
            label = n.get("label", "Unknown")
            by_label.setdefault(label, []).append(n)

        for label in _ALL_LABELS:
            node_list = by_label.get(label, [])
            if node_list:
                adapter.insert_nodes(node_list, label=label)

        adapter.insert_edges(edges)

        # Store IDs for queries
        self._person_ids = [n["id"] for n in by_label.get("Person", [])]
        self._city_ids = [n["id"] for n in by_label.get("City", [])]
        self._tag_ids = [n["id"] for n in by_label.get("Tag", [])]
        self._country_ids = [n["id"] for n in by_label.get("Country", [])]
        self._post_ids = [n["id"] for n in by_label.get("Post", [])]
        self._comment_ids = [n["id"] for n in by_label.get("Comment", [])]
        self._forum_ids = [n["id"] for n in by_label.get("Forum", [])]
        self._message_ids = self._post_ids + self._comment_ids
        self._person_count = len(self._person_ids)

    def _get_person(self, index: int = 0) -> str:
        if not self._person_ids:
            return "person_0"
        return self._person_ids[index % len(self._person_ids)]

    def _get_city(self, index: int = 0) -> str:
        if not self._city_ids:
            return "city_0"
        return self._city_ids[index % len(self._city_ids)]

    def _get_tag(self, index: int = 0) -> str:
        if not self._tag_ids:
            return "tag_0"
        return self._tag_ids[index % len(self._tag_ids)]

    def _get_message(self, index: int = 0) -> str:
        if not self._message_ids:
            return "post_0"
        return self._message_ids[index % len(self._message_ids)]

    def _get_post(self, index: int = 0) -> str:
        if not self._post_ids:
            return "post_0"
        return self._post_ids[index % len(self._post_ids)]

    def _get_forum(self, index: int = 0) -> str:
        if not self._forum_ids:
            return "forum_0"
        return self._forum_ids[index % len(self._forum_ids)]


# =============================================================================
# Short Reads (IS1-IS7)
# =============================================================================


@BenchmarkRegistry.register("snb_is1", category="ldbc_snb")
class SnbIS1Benchmark(SnbInteractiveBenchmarkBase):
    """IS1: Profile of a Person.

    Spec: Given personId, retrieve profile (firstName, lastName, birthday,
    locationIP, browserUsed, gender, creationDate) and the city they live in.
    """

    @property
    def name(self) -> str:
        return "snb_is1"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        found = 0
        for i in range(100):
            person_id = self._get_person(i)
            person = adapter.get_node(person_id)
            if not person:
                continue
            # Get city via LIVES_IN
            cities = adapter.get_neighbors(person_id, edge_type="LIVES_IN")
            if cities:
                adapter.get_node(cities[0])
            found += 1
        return found


@BenchmarkRegistry.register("snb_is2", category="ldbc_snb")
class SnbIS2Benchmark(SnbInteractiveBenchmarkBase):
    """IS2: Recent Messages of a Person.

    Spec: Given personId, find 10 most recent Messages (Post/Comment)
    created by that person. For each Comment, follow REPLY_OF chain to
    the original Post and retrieve the Post's creator.
    """

    @property
    def name(self) -> str:
        return "snb_is2"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        total = 0
        for i in range(20):
            person_id = self._get_person(i)
            # Person -[CREATED]-> Post/Comment (reverse helper edge)
            messages = adapter.get_neighbors(person_id, edge_type="CREATED")
            # Take top 10 (simulating ORDER BY creationDate DESC LIMIT 10)
            for msg_id in messages[:10]:
                msg = adapter.get_node(msg_id)
                if not msg:
                    continue
                total += 1
                # If it's a Comment, follow REPLY_OF to original Post
                if msg_id.startswith("comment_"):
                    current = msg_id
                    for _ in range(10):  # max depth
                        parents = adapter.get_neighbors(
                            current, edge_type="REPLY_OF",
                        )
                        if not parents:
                            break
                        current = parents[0]
                        if current.startswith("post_"):
                            # Get original post's creator
                            creators = adapter.get_neighbors(
                                current, edge_type="HAS_CREATOR",
                            )
                            if creators:
                                adapter.get_node(creators[0])
                            break
        return total


@BenchmarkRegistry.register("snb_is3", category="ldbc_snb")
class SnbIS3Benchmark(SnbInteractiveBenchmarkBase):
    """IS3: Friends of a Person.

    Spec: Given personId, retrieve all KNOWS friends with their
    properties (firstName, lastName) and friendship creationDate.
    """

    @property
    def name(self) -> str:
        return "snb_is3"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        total_friends = 0
        for i in range(50):
            person_id = self._get_person(i)
            friends = adapter.get_neighbors(person_id, edge_type="KNOWS")
            for friend_id in friends:
                adapter.get_node(friend_id)
            total_friends += len(friends)
        return total_friends


@BenchmarkRegistry.register("snb_is4", category="ldbc_snb")
class SnbIS4Benchmark(SnbInteractiveBenchmarkBase):
    """IS4: Content of a Message.

    Spec: Given messageId, retrieve its content/imageFile and creationDate.
    """

    @property
    def name(self) -> str:
        return "snb_is4"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        found = 0
        for i in range(100):
            msg_id = self._get_message(i)
            msg = adapter.get_node(msg_id)
            if msg and (msg.get("content") or msg.get("imageFile")):
                found += 1
        return found


@BenchmarkRegistry.register("snb_is5", category="ldbc_snb")
class SnbIS5Benchmark(SnbInteractiveBenchmarkBase):
    """IS5: Creator of a Message.

    Spec: Given messageId, find the Person who created it via HAS_CREATOR.
    Return personId, firstName, lastName.
    """

    @property
    def name(self) -> str:
        return "snb_is5"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        found = 0
        for i in range(50):
            msg_id = self._get_message(i)
            creators = adapter.get_neighbors(msg_id, edge_type="HAS_CREATOR")
            if creators:
                creator = adapter.get_node(creators[0])
                if creator:
                    found += 1
        return found


@BenchmarkRegistry.register("snb_is6", category="ldbc_snb")
class SnbIS6Benchmark(SnbInteractiveBenchmarkBase):
    """IS6: Forum of a Message.

    Spec: Given messageId, walk REPLY_OF chain to original Post,
    then find its Forum via IN_FORUM, then get the Forum's moderator
    via HAS_MODERATOR.
    """

    @property
    def name(self) -> str:
        return "snb_is6"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        found = 0
        for i in range(50):
            msg_id = self._get_message(i)
            # Walk REPLY_OF chain to root Post
            current = msg_id
            for _ in range(10):
                parents = adapter.get_neighbors(
                    current, edge_type="REPLY_OF",
                )
                if not parents:
                    break
                current = parents[0]
            # current is now the root Post (or original if it was a Post)
            # Post -[IN_FORUM]-> Forum
            forums = adapter.get_neighbors(current, edge_type="IN_FORUM")
            if not forums:
                continue
            forum = adapter.get_node(forums[0])
            if not forum:
                continue
            # Forum -[HAS_MODERATOR]-> Person
            moderators = adapter.get_neighbors(
                forums[0], edge_type="HAS_MODERATOR",
            )
            if moderators:
                adapter.get_node(moderators[0])
            found += 1
        return found


@BenchmarkRegistry.register("snb_is7", category="ldbc_snb")
class SnbIS7Benchmark(SnbInteractiveBenchmarkBase):
    """IS7: Replies of a Message.

    Spec: Given messageId, find all Comments that reply to it (HAS_REPLY).
    For each reply, get the author (HAS_CREATOR) and check whether the
    author KNOWS the original message's author.
    """

    @property
    def name(self) -> str:
        return "snb_is7"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        total_replies = 0
        for i in range(30):
            msg_id = self._get_message(i)
            # Get original message's creator
            orig_creators = adapter.get_neighbors(
                msg_id, edge_type="HAS_CREATOR",
            )
            orig_author = orig_creators[0] if orig_creators else None
            # Get replies via HAS_REPLY reverse helper
            replies = adapter.get_neighbors(msg_id, edge_type="HAS_REPLY")
            for reply_id in replies[:20]:
                reply = adapter.get_node(reply_id)
                if not reply:
                    continue
                # Get reply's creator
                reply_creators = adapter.get_neighbors(
                    reply_id, edge_type="HAS_CREATOR",
                )
                if reply_creators and orig_author:
                    # Check if reply author KNOWS original author
                    reply_author_friends = adapter.get_neighbors(
                        reply_creators[0], edge_type="KNOWS",
                    )
                    _ = orig_author in reply_author_friends
                total_replies += 1
        return total_replies


# =============================================================================
# Complex Reads (IC1, IC2, IC3, IC6)
# =============================================================================


@BenchmarkRegistry.register("snb_ic1", category="ldbc_snb")
class SnbIC1Benchmark(SnbInteractiveBenchmarkBase):
    """IC1: Transitive Friends with Certain Name.

    Spec: Given personId and firstName, BFS up to 3 hops via KNOWS,
    filter persons by firstName, return profile + city + universities +
    companies. LIMIT 20.
    """

    @property
    def name(self) -> str:
        return "snb_ic1"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        total_matches = 0
        for i in range(10):
            person_id = self._get_person(i)
            # Deterministic target name for filtering
            target_name = FIRST_NAMES_MALE[i % len(FIRST_NAMES_MALE)]
            # BFS up to 3 hops via KNOWS
            visited = adapter.traverse_bfs(
                person_id, max_depth=3, edge_type="KNOWS",
            )
            matches = 0
            for vid in visited:
                if matches >= 20:
                    break
                node = adapter.get_node(vid)
                if not node:
                    continue
                if node.get("firstName") == target_name:
                    # Get city via LIVES_IN
                    adapter.get_neighbors(vid, edge_type="LIVES_IN")
                    # Get universities via STUDY_AT
                    adapter.get_neighbors(vid, edge_type="STUDY_AT")
                    # Get companies via WORK_AT
                    adapter.get_neighbors(vid, edge_type="WORK_AT")
                    matches += 1
            total_matches += matches
        return total_matches


@BenchmarkRegistry.register("snb_ic2", category="ldbc_snb")
class SnbIC2Benchmark(SnbInteractiveBenchmarkBase):
    """IC2: Recent Messages by Your Friends.

    Spec: Given personId, find friends via KNOWS, get their messages
    via CREATED edge, sort by creationDate DESC, LIMIT 20.
    """

    @property
    def name(self) -> str:
        return "snb_ic2"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        total = 0
        for i in range(10):
            person_id = self._get_person(i)
            friends = adapter.get_neighbors(person_id, edge_type="KNOWS")
            all_messages: list[dict[str, Any]] = []
            for friend_id in friends[:20]:
                messages = adapter.get_neighbors(
                    friend_id, edge_type="CREATED",
                )
                for msg_id in messages[:5]:
                    msg = adapter.get_node(msg_id)
                    if msg:
                        msg["_id"] = msg_id
                        msg["_friend"] = friend_id
                        all_messages.append(msg)
            # Sort by creationDate DESC, take 20
            all_messages.sort(
                key=lambda m: m.get("creationDate", ""), reverse=True,
            )
            total += len(all_messages[:20])
        return total


@BenchmarkRegistry.register("snb_ic3", category="ldbc_snb")
class SnbIC3Benchmark(SnbInteractiveBenchmarkBase):
    """IC3: Friends and Friends-of-Friends in Countries.

    Spec: Given personId and two countries, find 1-2 hop KNOWS friends
    NOT in those countries, count their messages located in each country.
    """

    @property
    def name(self) -> str:
        return "snb_ic3"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        total = 0
        for i in range(5):
            person_id = self._get_person(i)
            # 1-hop friends
            friends_1 = adapter.get_neighbors(
                person_id, edge_type="KNOWS",
            )
            candidates = set(friends_1)
            # 2-hop friends (limited)
            for f in friends_1[:10]:
                fof = adapter.get_neighbors(f, edge_type="KNOWS")
                candidates.update(fof[:5])
            candidates.discard(person_id)
            for cid in list(candidates)[:20]:
                # Check person's location (city -> country)
                cities = adapter.get_neighbors(cid, edge_type="LIVES_IN")
                if cities:
                    adapter.get_neighbors(
                        cities[0], edge_type="IS_PART_OF",
                    )
                # Count messages in target countries
                messages = adapter.get_neighbors(cid, edge_type="CREATED")
                for msg_id in messages[:5]:
                    loc = adapter.get_neighbors(
                        msg_id, edge_type="IS_LOCATED_IN",
                    )
                    total += len(loc)
        return total


@BenchmarkRegistry.register("snb_ic6", category="ldbc_snb")
class SnbIC6Benchmark(SnbInteractiveBenchmarkBase):
    """IC6: Tag Co-occurrence.

    Spec: Given personId and tagId, find 2-hop KNOWS friends, get their
    Posts with the given Tag, count co-occurring Tags on those Posts.
    """

    @property
    def name(self) -> str:
        return "snb_ic6"

    def run_iteration(
        self, adapter: GraphDatabaseAdapter, scale: ScaleConfig,
    ) -> int:
        total = 0
        for i in range(10):
            person_id = self._get_person(i)
            tag_id = self._get_tag(i)
            # 1-hop + 2-hop KNOWS friends
            friends = adapter.get_neighbors(
                person_id, edge_type="KNOWS",
            )
            fof = set(friends)
            for f in friends[:10]:
                fof.update(
                    adapter.get_neighbors(f, edge_type="KNOWS")[:5],
                )
            fof.discard(person_id)
            # For each friend, get their posts and check tags
            tag_counts: dict[str, int] = {}
            for pid in list(fof)[:20]:
                posts = adapter.get_neighbors(pid, edge_type="CREATED")
                for post_id in posts[:5]:
                    if not post_id.startswith("post_"):
                        continue
                    post_tags = adapter.get_neighbors(
                        post_id, edge_type="HAS_TAG",
                    )
                    if tag_id in post_tags:
                        for other_tag in post_tags:
                            if other_tag != tag_id:
                                tag_counts[other_tag] = (
                                    tag_counts.get(other_tag, 0) + 1
                                )
            total += len(tag_counts)
        return total
