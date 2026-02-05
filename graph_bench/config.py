r"""
Benchmark configuration and scale definitions.

LDBC SNB Scale Factors (official):
    - sf01: SF0.1 - 1K persons, 18K edges (quick validation)
    - sf1: SF1 - 10K persons, 180K edges (standard)
    - sf3: SF3 - 27K persons, 540K edges
    - sf10: SF10 - 73K persons, 2M edges
    - sf30: SF30 - 180K persons, 6.5M edges
    - sf100: SF100 - 280K persons, 18M edges (full scale)

Reference: https://ldbcouncil.org/benchmarks/snb/
Spec: https://ldbcouncil.org/ldbc_snb_docs/ldbc-snb-specification.pdf

    from graph_bench.config import SCALES, get_scale

    scale = get_scale("sf1")  # Standard LDBC SF1
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

# LDBC SNB Scale Factors
# Reference: https://ldbcouncil.org/ldbc_snb_docs/ldbc-snb-specification.pdf
# Person counts and KNOWS edge counts from official LDBC datagen
SCALES: dict[str, ScaleConfig] = {
    # SF0.1 - Quick validation
    "sf01": ScaleConfig(
        name="sf01",
        nodes=1_000,
        edges=18_000,
        warmup_iterations=2,
        measurement_iterations=5,
        timeout_seconds=60,
    ),
    # SF1 - Standard benchmark (default)
    "sf1": ScaleConfig(
        name="sf1",
        nodes=10_000,
        edges=180_000,
        warmup_iterations=3,
        measurement_iterations=10,
        timeout_seconds=120,
    ),
    # SF3
    "sf3": ScaleConfig(
        name="sf3",
        nodes=27_000,
        edges=540_000,
        warmup_iterations=3,
        measurement_iterations=10,
        timeout_seconds=300,
    ),
    # SF10
    "sf10": ScaleConfig(
        name="sf10",
        nodes=73_000,
        edges=2_000_000,
        warmup_iterations=5,
        measurement_iterations=10,
        timeout_seconds=600,
    ),
    # SF30
    "sf30": ScaleConfig(
        name="sf30",
        nodes=180_000,
        edges=6_500_000,
        warmup_iterations=5,
        measurement_iterations=10,
        timeout_seconds=1800,
    ),
    # SF100 - Full scale (reduced iterations due to long query times)
    "sf100": ScaleConfig(
        name="sf100",
        nodes=280_000,
        edges=18_000_000,
        warmup_iterations=1,
        measurement_iterations=1,
        timeout_seconds=3600,
    ),
}

# Aliases for backward compatibility
SCALES["small"] = SCALES["sf01"]
SCALES["medium"] = SCALES["sf1"]
SCALES["large"] = SCALES["sf100"]

DEFAULT_SCALE = "sf1"


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
