"""Memory measurement utilities for benchmarks.

Provides functions to measure memory usage for both embedded databases
(via Python process RSS) and Docker-based databases (via docker stats).
"""

from __future__ import annotations

import os
import subprocess

__all__ = [
    "get_process_memory",
    "get_container_memory",
    "CONTAINER_NAMES",
]

# Mapping of adapter names to Docker container names
CONTAINER_NAMES: dict[str, str] = {
    "Neo4j": "graph-bench-neo4j",
    "Memgraph": "graph-bench-memgraph",
    "ArangoDB": "graph-bench-arangodb",
    "FalkorDB": "graph-bench-falkordb",
    "NebulaGraph": "graph-bench-nebula-graphd",
}


def get_process_memory() -> int:
    """Get Python process RSS memory in bytes.

    Uses psutil if available, otherwise returns 0.

    Returns:
        Resident Set Size (RSS) in bytes, or 0 if unavailable.
    """
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        return 0
    except Exception:
        return 0


def get_container_memory(adapter_name: str) -> int:
    """Get Docker container memory usage in bytes.

    Uses docker stats CLI to query container memory.

    Args:
        adapter_name: Name of the adapter (e.g., "Neo4j", "Memgraph").

    Returns:
        Memory usage in bytes, or 0 if unavailable.
    """
    container = CONTAINER_NAMES.get(adapter_name)
    if not container:
        return 0

    try:
        # Use --context desktop-linux for Docker Desktop on Windows
        result = subprocess.run(
            [
                "docker",
                "--context",
                "desktop-linux",
                "stats",
                container,
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            # Try without context flag
            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    container,
                    "--no-stream",
                    "--format",
                    "{{.MemUsage}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        return _parse_docker_memory(result.stdout.strip())
    except Exception:
        return 0


def _parse_docker_memory(mem_str: str) -> int:
    """Parse Docker memory string to bytes.

    Handles formats like '123.4MiB / 8GiB' or '123.4MB / 8GB'.

    Args:
        mem_str: Memory string from docker stats.

    Returns:
        Memory in bytes, or 0 if parsing fails.
    """
    if not mem_str or "/" not in mem_str:
        return 0

    mem_part = mem_str.split("/")[0].strip()

    # Handle both IEC (MiB) and SI (MB) units
    units = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "kB": 1000,
    }

    for unit, mult in sorted(units.items(), key=lambda x: -len(x[0])):
        if mem_part.endswith(unit):
            try:
                value = float(mem_part[: -len(unit)])
                return int(value * mult)
            except ValueError:
                return 0

    return 0
