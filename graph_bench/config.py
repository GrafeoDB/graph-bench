r"""
Benchmark configuration and scale definitions.

Scales:
    - small: Quick validation, ~10K nodes
    - medium: Standard benchmark, ~100K nodes
    - large: Full scale, ~1M nodes

    from graph_bench.config import SCALES, get_scale

    scale = get_scale("medium")
    print(f"Nodes: {scale.nodes}, Edges: {scale.edges}")
"""

import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv

    # Look for .env in current dir or parent dirs
    env_file = Path(".env")
    if not env_file.exists():
        env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not installed, skip

from graph_bench.types import ScaleConfig

__all__ = [
    "SCALES",
    "DEFAULT_SCALE",
    "get_scale",
    "get_env",
    "ENV_PREFIX",
]

ENV_PREFIX = "GRAPH_BENCH_"

SCALES: dict[str, ScaleConfig] = {
    "small": ScaleConfig(
        name="small",
        nodes=10_000,
        edges=50_000,
        warmup_iterations=2,
        measurement_iterations=5,
        timeout_seconds=60,
    ),
    "medium": ScaleConfig(
        name="medium",
        nodes=100_000,
        edges=500_000,
        warmup_iterations=3,
        measurement_iterations=10,
        timeout_seconds=300,
    ),
    "large": ScaleConfig(
        name="large",
        nodes=1_000_000,
        edges=5_000_000,
        warmup_iterations=5,
        measurement_iterations=10,
        timeout_seconds=1800,
    ),
}

DEFAULT_SCALE = "medium"


def get_scale(name: str) -> ScaleConfig:
    """Get scale configuration by name.

    Args:
        name: Scale name (small, medium, large).

    Returns:
        ScaleConfig for the requested scale.

    Raises:
        ValueError: If scale name is not recognized.
    """
    if name not in SCALES:
        valid = ", ".join(SCALES.keys())
        msg = f"Unknown scale '{name}'. Valid scales: {valid}"
        raise ValueError(msg)
    return SCALES[name]


def get_env(key: str, *, default: str | None = None) -> str | None:
    """Get environment variable with GRAPH_BENCH_ prefix.

    Args:
        key: Variable name without prefix (e.g., "NEO4J_URI").
        default: Default value if not set.

    Returns:
        Environment variable value or default.
    """
    return os.environ.get(f"{ENV_PREFIX}{key}", default)
