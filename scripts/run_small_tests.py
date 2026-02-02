#!/usr/bin/env python
"""Run small tests on all database adapters."""

from graph_bench.adapters import AdapterRegistry

# Connection configs for each database
CONFIGS = {
    # Embedded databases (no server needed)
    "ladybug": {"uri": ":memory:"},
    "duckdb": {"uri": ":memory:"},
    "grafeo": {},  # In-memory by default

    # Docker databases
    "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "benchmark"},
    "memgraph": {"uri": "bolt://localhost:7688"},
    "arangodb": {"uri": "http://localhost:8529", "user": "root", "password": "benchmark"},
}

DATABASE_ORDER = ["ladybug", "duckdb", "grafeo", "neo4j", "memgraph", "arangodb"]


def test_adapter(name: str) -> tuple[bool, str]:
    """Run small test on an adapter. Returns (success, message)."""
    try:
        config = CONFIGS.get(name, {})
        adapter = AdapterRegistry.create(name)
        adapter.connect(**config)

        # Clear any existing data
        adapter.clear()

        # Test 1: Insert nodes
        nodes = [{"id": f"n{i}", "name": f"Node{i}", "value": i} for i in range(10)]
        node_count = adapter.insert_nodes(nodes, label="TestNode")
        if node_count != 10:
            return False, f"Expected 10 nodes inserted, got {node_count}"

        # Verify count
        count = adapter.count_nodes()
        if count != 10:
            return False, f"Expected 10 nodes, count returned {count}"

        # Test 2: Get single node
        node = adapter.get_node("n0")
        if node is None:
            return False, "Failed to get node n0"
        if node.get("id") != "n0":
            return False, f"Node id mismatch: {node}"

        # Test 3: Insert edges (chain: n0->n1->n2->...->n9)
        edges = [(f"n{i}", f"n{i+1}", "NEXT", {"order": i}) for i in range(9)]
        edge_count = adapter.insert_edges(edges)
        if edge_count != 9:
            return False, f"Expected 9 edges inserted, got {edge_count}"

        # Verify edge count
        count = adapter.count_edges()
        if count != 9:
            return False, f"Expected 9 edges, count returned {count}"

        # Test 4: Get neighbors
        neighbors = adapter.get_neighbors("n0")
        if "n1" not in neighbors:
            return False, f"Expected n1 in neighbors of n0, got {neighbors}"

        # Test 5: Shortest path
        path = adapter.shortest_path("n0", "n5")
        if path is None:
            return False, "Shortest path returned None"
        if path != ["n0", "n1", "n2", "n3", "n4", "n5"]:
            return False, f"Unexpected path: {path}"

        # Test 6: BFS traversal
        bfs_nodes = adapter.traverse_bfs("n0", max_depth=3)
        expected_bfs = {"n0", "n1", "n2", "n3"}
        if set(bfs_nodes) != expected_bfs:
            return False, f"BFS expected {expected_bfs}, got {set(bfs_nodes)}"

        # Test 7: DFS traversal
        dfs_nodes = adapter.traverse_dfs("n0", max_depth=3)
        if set(dfs_nodes) != expected_bfs:
            return False, f"DFS expected {expected_bfs}, got {set(dfs_nodes)}"

        # Clean up
        adapter.clear()
        adapter.disconnect()

        return True, "All 7 tests passed"

    except ImportError as e:
        return False, f"Missing package: {e}"
    except Exception as e:
        return False, str(e)[:80]


def main():
    print("=" * 70)
    print("Small Database Tests")
    print("=" * 70)
    print()
    print("Testing 7 operations: insert_nodes, get_node, insert_edges,")
    print("                      get_neighbors, shortest_path, BFS, DFS")
    print()

    results = {}
    for name in DATABASE_ORDER:
        print(f"Testing {name}...", end=" ", flush=True)
        ok, msg = test_adapter(name)
        results[name] = (ok, msg)
        status = "[OK]" if ok else "[FAIL]"
        print(f"{status}")
        if not ok:
            print(f"  -> {msg}")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    passed = sum(1 for ok, _ in results.values() if ok)
    failed = len(results) - passed
    print(f"Passed: {passed}/{len(results)}")
    if failed > 0:
        print(f"Failed: {failed}")
        for name, (ok, msg) in results.items():
            if not ok:
                print(f"  - {name}: {msg}")
    print("=" * 70)


if __name__ == "__main__":
    main()
