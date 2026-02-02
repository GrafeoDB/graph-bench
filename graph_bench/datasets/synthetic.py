r"""
Synthetic social network dataset generator.

Generates a social network graph with Person nodes and FOLLOWS edges.
Optionally uses Faker for realistic names.

    from graph_bench.datasets.synthetic import SyntheticSocialNetwork

    dataset = SyntheticSocialNetwork()
    nodes, edges = dataset.generate(SCALES["medium"])
"""

import random
from typing import Any

from graph_bench.datasets.base import BaseDatasetLoader
from graph_bench.types import ScaleConfig

__all__ = ["SyntheticSocialNetwork"]


class SyntheticSocialNetwork(BaseDatasetLoader):
    """Synthetic social network dataset generator."""

    def __init__(self, *, seed: int | None = None, use_faker: bool = True) -> None:
        """Initialize generator.

        Args:
            seed: Random seed for reproducibility.
            use_faker: Use Faker for realistic names (requires faker package).
        """
        self._seed = seed
        self._use_faker = use_faker
        self._faker: Any = None

    @property
    def name(self) -> str:
        return "synthetic_social"

    def _init_faker(self) -> None:
        if self._faker is not None:
            return

        if not self._use_faker:
            return

        try:
            from faker import Faker

            self._faker = Faker()
            if self._seed is not None:
                Faker.seed(self._seed)
        except ImportError:
            self._use_faker = False

    def generate(self, scale: ScaleConfig) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, dict[str, Any]]]]:
        """Generate synthetic social network.

        Args:
            scale: Scale configuration determining node/edge counts.

        Returns:
            Tuple of (nodes, edges).
        """
        if self._seed is not None:
            random.seed(self._seed)

        self._init_faker()

        nodes = self._generate_nodes(scale.nodes)
        edges = self._generate_edges(scale.nodes, scale.edges)

        return nodes, edges

    def _generate_nodes(self, count: int) -> list[dict[str, Any]]:
        """Generate Person nodes."""
        nodes = []
        cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
        ]

        for i in range(count):
            if self._faker:
                node = {
                    "id": f"person_{i}",
                    "name": self._faker.name(),
                    "age": random.randint(18, 80),
                    "city": random.choice(cities),
                    "email": self._faker.email(),
                }
            else:
                node = {
                    "id": f"person_{i}",
                    "name": f"Person {i}",
                    "age": 20 + (i % 60),
                    "city": cities[i % len(cities)],
                    "email": f"person{i}@example.com",
                }
            nodes.append(node)

        return nodes

    def _generate_edges(self, node_count: int, edge_count: int) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate FOLLOWS edges with power-law distribution."""
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        edge_set: set[tuple[str, str]] = set()

        popular_nodes = list(range(min(100, node_count // 10)))

        attempts = 0
        max_attempts = edge_count * 3

        while len(edges) < edge_count and attempts < max_attempts:
            attempts += 1

            if random.random() < 0.3 and popular_nodes:
                tgt_idx = random.choice(popular_nodes)
                src_idx = random.randint(0, node_count - 1)
            else:
                src_idx = random.randint(0, node_count - 1)
                tgt_idx = random.randint(0, node_count - 1)

            if src_idx == tgt_idx:
                continue

            src = f"person_{src_idx}"
            tgt = f"person_{tgt_idx}"

            if (src, tgt) in edge_set:
                continue

            edge_set.add((src, tgt))
            props = {"since": random.randint(2015, 2025), "weight": round(random.random(), 2)}
            edges.append((src, tgt, "FOLLOWS", props))

        return edges
