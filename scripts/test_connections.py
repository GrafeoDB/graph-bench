#!/usr/bin/env python
"""Test connections to all database adapters."""

from graph_bench.adapters import AdapterRegistry

# Connection configs for each database
CONFIGS = {
    # Embedded databases (no server needed)
    "ladybug": {"uri": ":memory:"},
    "grafeo": {},  # In-memory by default

    # Docker databases
    "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "benchmark"},
    "memgraph": {"uri": "bolt://localhost:7688"},
    "arangodb": {"uri": "http://localhost:8529", "user": "root", "password": "benchmark"},
    "tugraph": {"uri": "bolt://localhost:7689", "user": "admin", "password": "73@TuGraph"},
}


def test_connection(name: str) -> tuple[bool, str]:
    """Test connection to a database. Returns (success, message)."""
    try:
        config = CONFIGS.get(name, {})
        adapter = AdapterRegistry.create(name)
        adapter.connect(**config)
        version = adapter.version
        adapter.disconnect()
        return True, f"v{version}"
    except ImportError as e:
        return False, f"Missing package: {e}"
    except Exception as e:
        return False, str(e)[:50]


def main():
    print("=" * 60)
    print("Testing Database Connections")
    print("=" * 60)
    print()

    # Test embedded first (don't need Docker)
    print("EMBEDDED DATABASES (no Docker needed)")
    print("-" * 40)
    for name in ["ladybug", "grafeo"]:
        ok, msg = test_connection(name)
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status:6} {name:12} {msg}")

    print()
    print("DOCKER DATABASES (need docker compose up)")
    print("-" * 40)
    for name in ["neo4j", "memgraph", "arangodb", "tugraph"]:
        ok, msg = test_connection(name)
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status:6} {name:12} {msg}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
