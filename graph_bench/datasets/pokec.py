r"""
Pokec Social Network dataset.

Pokec is the most popular Slovak social network. The dataset contains
anonymized data of the whole network from 2012, with 1.6M users and
30.6M friendship edges.

This generator creates synthetic data following Pokec's schema and
statistical properties, useful for benchmarking without downloading
the full dataset.

Schema:
    User -[FRIEND]-> User (directed friendships)

Properties per User:
    - id, public (visibility)
    - gender, age, region
    - registration date, last login
    - body features (height, weight, BMI)
    - life_goals, relationship status
    - interests (I_* columns)

Statistics (full dataset):
    - Nodes: 1,632,803 users
    - Edges: 30,622,564 friendships
    - Avg degree: ~37.5
    - Diameter: 11

References:
    - https://snap.stanford.edu/data/soc-pokec.html
    - Takac, Zabovsky: "Data Analysis in Public Social Networks" (2012)

    from graph_bench.datasets.pokec import PokecSocialNetwork

    dataset = PokecSocialNetwork()
    nodes, edges = dataset.generate(scale)
"""

import random
from datetime import datetime, timedelta
from typing import Any

from graph_bench.datasets.base import BaseDatasetLoader
from graph_bench.types import ScaleConfig

__all__ = ["PokecSocialNetwork"]

# Slovak regions (Pokec is Slovak)
REGIONS = [
    "Bratislavský", "Trnavský", "Trenčiansky", "Nitriansky",
    "Žilinský", "Banskobystrický", "Prešovský", "Košický",
]

# Relationship statuses (Slovak social network context)
RELATIONSHIP_STATUS = [
    "single", "in_relationship", "engaged", "married",
    "complicated", "divorced", "widowed", None,
]

# Interest categories (Pokec I_* columns)
INTEREST_CATEGORIES = [
    "movies", "music", "books", "sports", "travel", "art",
    "technology", "gaming", "cooking", "fashion", "nature",
    "politics", "science", "photography", "cars", "pets",
]

# Life goals
LIFE_GOALS = [
    "career", "family", "travel", "education", "health",
    "creativity", "wealth", "happiness", "adventure", None,
]


class PokecSocialNetwork(BaseDatasetLoader):
    """Pokec Social Network dataset generator.

    Generates a social network graph following Pokec's schema and
    statistical properties including directed friendships, user
    demographics, and interest profiles.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        avg_degree: float = 37.5,
        include_interests: bool = True,
    ) -> None:
        """Initialize Pokec dataset generator.

        Args:
            seed: Random seed for reproducibility.
            avg_degree: Average node degree (Pokec real: ~37.5).
            include_interests: Include interest columns (I_*).
        """
        self._seed = seed
        self._avg_degree = avg_degree
        self._include_interests = include_interests

    @property
    def name(self) -> str:
        return "pokec"

    def generate(
        self, scale: ScaleConfig
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, dict[str, Any]]]]:
        """Generate Pokec-like social network dataset."""
        if self._seed is not None:
            random.seed(self._seed)

        node_count = scale.nodes
        # Calculate edges to match avg degree, capped by scale config
        target_edges = int(node_count * self._avg_degree / 2)
        edge_count = min(scale.edges, target_edges)

        # Generate user nodes
        nodes = self._generate_users(node_count)

        # Generate directed friendship edges
        edges = self._generate_friendships(node_count, edge_count)

        return nodes, edges

    def _generate_users(self, count: int) -> list[dict[str, Any]]:
        """Generate User nodes with Pokec-realistic properties."""
        users = []
        registration_start = datetime(2004, 1, 1)  # Pokec launched ~2004
        registration_end = datetime(2012, 1, 1)  # Dataset snapshot

        for i in range(count):
            # Registration and last login
            reg_days = random.randint(0, (registration_end - registration_start).days)
            registration = registration_start + timedelta(days=reg_days)
            last_login_offset = random.randint(0, 365)
            last_login = registration + timedelta(days=last_login_offset)

            # Demographics
            gender = random.choice([0, 1, None])  # 0=female, 1=male, None=unknown
            age = random.randint(13, 70) if random.random() > 0.1 else None

            # Body features (many users don't fill these)
            height = random.randint(150, 200) if random.random() > 0.5 else None
            weight = random.randint(45, 120) if random.random() > 0.5 else None
            bmi = round(weight / ((height / 100) ** 2), 1) if height and weight else None

            user: dict[str, Any] = {
                "id": f"user_{i}",
                "label": "User",
                "public": random.choice([0, 1]),  # Profile visibility
                "gender": gender,
                "age": age,
                "region": random.choice(REGIONS),
                "registration": registration.strftime("%Y-%m-%d"),
                "last_login": last_login.strftime("%Y-%m-%d %H:%M:%S"),
                "height": height,
                "weight": weight,
                "BMI": bmi,
                "relationship": random.choice(RELATIONSHIP_STATUS),
                "life_goal": random.choice(LIFE_GOALS),
            }

            # Add interest columns (I_*)
            if self._include_interests:
                for category in INTEREST_CATEGORIES:
                    # Boolean or intensity (0-5) for each interest
                    if random.random() > 0.7:  # 30% chance of having each interest
                        user[f"I_{category}"] = random.randint(1, 5)
                    else:
                        user[f"I_{category}"] = 0

            users.append(user)

        return users

    def _generate_friendships(
        self, user_count: int, edge_count: int
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate directed FRIEND edges with power-law distribution.

        Pokec has directed friendships - A following B doesn't mean B follows A.
        The network exhibits power-law degree distribution.
        """
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        edge_set: set[tuple[int, int]] = set()

        # Power-law parameters
        # Some users are "popular" with many followers
        popular_fraction = 0.01
        popular_count = max(10, int(user_count * popular_fraction))
        popular_users = list(range(popular_count))

        attempts = 0
        max_attempts = edge_count * 4

        while len(edges) < edge_count and attempts < max_attempts:
            attempts += 1

            # Sampling strategy for power-law distribution
            r = random.random()

            if r < 0.3 and popular_users:
                # 30%: Connect TO a popular user (follower -> popular)
                src = random.randint(0, user_count - 1)
                tgt = random.choice(popular_users)
            elif r < 0.5 and popular_users:
                # 20%: Connect FROM a popular user (popular -> follower)
                src = random.choice(popular_users)
                tgt = random.randint(0, user_count - 1)
            elif r < 0.7:
                # 20%: Local connections (nearby IDs often know each other)
                src = random.randint(0, user_count - 1)
                offset = random.randint(1, min(100, user_count // 10))
                tgt = (src + offset) % user_count
            else:
                # 30%: Random connections
                src = random.randint(0, user_count - 1)
                tgt = random.randint(0, user_count - 1)

            # No self-loops
            if src == tgt:
                continue

            # Check for duplicates (directed, so (a,b) != (b,a))
            if (src, tgt) in edge_set:
                continue

            edge_set.add((src, tgt))
            edges.append((f"user_{src}", f"user_{tgt}", "FRIEND", {}))

        return edges
