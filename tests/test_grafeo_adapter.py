r"""
Embedded Grafeo adapter: storage modes, native paths and regressions.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from graph_bench.adapters.base import BaseAdapter
from graph_bench.adapters.grafeo import GrafeoAdapter

from .grafeo_dataset import FOLLOWS, PEOPLE, load


@pytest.fixture
def adapter() -> Iterator[GrafeoAdapter]:
    adapter = GrafeoAdapter()
    adapter.connect(path=":memory:")
    yield adapter
    adapter.disconnect()


def test_in_memory_by_default(adapter: GrafeoAdapter) -> None:
    assert adapter.path is None
    assert adapter.connected
    assert adapter.is_embedded


def test_version_is_the_installed_grafeo() -> None:
    import grafeo

    assert GrafeoAdapter().version == grafeo.__version__


def test_file_database_is_a_single_grafeo_file(tmp_path: Path) -> None:
    # grafeo 0.6 deprecates the WAL-directory layout, so a path without the
    # suffix becomes a `.grafeo` file instead of a directory.
    adapter = GrafeoAdapter()
    adapter.connect(path=str(tmp_path / "data" / "bench"))
    assert adapter.path == tmp_path / "data" / "bench.grafeo"
    adapter.insert_nodes([{"id": "a"}])
    adapter.disconnect()
    assert (tmp_path / "data" / "bench.grafeo").is_file()
    assert not (tmp_path / "data" / "bench").exists()


def test_clear_keeps_a_file_database_on_disk(tmp_path: Path) -> None:
    # Regression: clear() used to swap a file database for an in-memory one.
    path = tmp_path / "bench.grafeo"
    adapter = GrafeoAdapter()
    adapter.connect(path=str(path))
    load(adapter)
    adapter.clear()
    assert adapter.count_nodes() == 0
    assert adapter.path == path
    adapter.insert_nodes([{"id": "kept", "name": "Kept"}], label="Person")
    adapter.disconnect()

    # disconnect() closes the database, so the same file opens again.
    reopened = GrafeoAdapter()
    reopened.connect(path=str(path))
    assert reopened.count_nodes() == 1
    assert reopened.get_node("kept") == {"id": "kept", "name": "Kept"}
    reopened.disconnect()


def test_insert_nodes_creates_the_id_index_once(adapter: GrafeoAdapter) -> None:
    adapter.insert_nodes(PEOPLE[:3], label="Person")
    assert adapter._db.has_property_index("id")
    adapter.insert_nodes(PEOPLE[3:], label="Person")
    assert adapter.count_nodes(label="Person") == len(PEOPLE)


def test_insert_nodes_respects_batch_size(adapter: GrafeoAdapter, monkeypatch: pytest.MonkeyPatch) -> None:
    batches: list[int] = []
    real_db = adapter._db

    class Recording:
        def batch_create_nodes_with_props(self, labels: Any, rows: list[dict[str, Any]]) -> list[int]:
            batches.append(len(rows))
            return real_db.batch_create_nodes_with_props(labels, rows)

        def __getattr__(self, name: str) -> Any:
            return getattr(real_db, name)

    monkeypatch.setattr(adapter, "_db", Recording())
    nodes = [{"id": f"n{i}"} for i in range(25)]
    assert adapter.insert_nodes(nodes, batch_size=10) == 25
    assert batches == [10, 10, 5]


def test_traverse_bfs_runs_natively(adapter: GrafeoAdapter, monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: the native BFS was called with an argument 0.5.44 rejects,
    # and the error was swallowed, so every BFS silently ran the fallback.
    load(adapter)

    def no_fallback(*args: Any, **kwargs: Any) -> list[str]:
        raise AssertionError("traverse_bfs used the get_neighbors fallback")

    monkeypatch.setattr(adapter, "get_neighbors", no_fallback)
    assert set(adapter.traverse_bfs("p0", max_depth=1)) == {"p0", "p1", "p3", "c0"}


def test_native_errors_are_not_swallowed(adapter: GrafeoAdapter, monkeypatch: pytest.MonkeyPatch) -> None:
    from grafeo import GrafeoError

    load(adapter)
    real_db = adapter._db

    class Failing:
        @property
        def algorithms(self) -> Any:
            raise GrafeoError("simulated engine failure")

        def __getattr__(self, name: str) -> Any:
            return getattr(real_db, name)

    monkeypatch.setattr(adapter, "_db", Failing())
    with pytest.raises(GrafeoError, match="simulated"):
        adapter.traverse_bfs("p0", max_depth=2)
    with pytest.raises(GrafeoError, match="simulated"):
        adapter.shortest_path("p0", "p5")


def test_weighted_shortest_path(adapter: GrafeoAdapter) -> None:
    # Regression: the weighted path came back as [str(distance), str(path)].
    load(adapter)
    assert adapter.shortest_path("p0", "p3") == ["p0", "p3"]
    assert adapter.shortest_path("p0", "p3", weighted=True) == ["p0", "p1", "p2", "p3"]
    assert adapter.shortest_path("p0", "nobody") is None


def test_sssp_is_keyed_by_benchmark_id(adapter: GrafeoAdapter) -> None:
    # Regression: algorithms.sssp() keys distances by the `name` property.
    load(adapter)
    distances = adapter.sssp("p0")
    assert set(distances) <= {node["id"] for node in PEOPLE} | {"c0", "c1"}
    assert distances["p1"] == 1.0
    assert distances["p3"] == 7.0


def test_louvain_returns_node_sets(adapter: GrafeoAdapter) -> None:
    # Regression: louvain() returns a dict, and iterating it gave sets of characters.
    load(adapter)
    communities = adapter.community_detection(algorithm="louvain")
    members = {node for community in communities for node in community}
    assert members == {node["id"] for node in PEOPLE} | {"c0", "c1"}


def test_unknown_start_node_uses_the_fallback(adapter: GrafeoAdapter) -> None:
    load(adapter)
    assert adapter.traverse_bfs("nobody", max_depth=2) == ["nobody"]


def test_edges_follow_their_direction(adapter: BaseAdapter) -> None:
    # The test graph has no pair of opposite edges.
    load(adapter)
    for src, tgt, _ in FOLLOWS:
        assert tgt in adapter.get_neighbors(src, edge_type="FOLLOWS")
        assert src not in adapter.get_neighbors(tgt, edge_type="FOLLOWS")
