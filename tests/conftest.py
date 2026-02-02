r"""
Shared pytest fixtures for graph-bench tests.
"""

import pytest

from graph_bench.config import SCALES
from graph_bench.types import ScaleConfig


@pytest.fixture
def small_scale() -> ScaleConfig:
    """Small scale configuration for testing."""
    return SCALES["small"]


@pytest.fixture
def tiny_scale() -> ScaleConfig:
    """Tiny scale for fast unit tests."""
    return ScaleConfig(
        name="tiny",
        nodes=100,
        edges=200,
        warmup_iterations=1,
        measurement_iterations=2,
        timeout_seconds=10,
    )


@pytest.fixture
def sample_nodes() -> list[dict]:
    """Sample node data for testing."""
    return [
        {"id": "person_0", "name": "Alice", "age": 30},
        {"id": "person_1", "name": "Bob", "age": 25},
        {"id": "person_2", "name": "Charlie", "age": 35},
        {"id": "person_3", "name": "Diana", "age": 28},
        {"id": "person_4", "name": "Eve", "age": 32},
    ]


@pytest.fixture
def sample_edges() -> list[tuple]:
    """Sample edge data for testing."""
    return [
        ("person_0", "person_1", "FOLLOWS", {"since": 2020}),
        ("person_0", "person_2", "FOLLOWS", {"since": 2021}),
        ("person_1", "person_2", "FOLLOWS", {"since": 2019}),
        ("person_2", "person_3", "FOLLOWS", {"since": 2022}),
        ("person_3", "person_4", "FOLLOWS", {"since": 2020}),
        ("person_4", "person_0", "FOLLOWS", {"since": 2021}),
    ]
