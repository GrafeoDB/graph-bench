r"""
The vector benchmark families: scan (brute force, every adapter) and ANN (native only).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest

from graph_bench.adapters.grafeo import GrafeoAdapter
from graph_bench.benchmarks import BenchmarkRegistry
from graph_bench.benchmarks.vector import (
    AnnSettings,
    VectorAnnDeleteBenchmark,
    VectorAnnKnnBenchmark,
    ann_settings,
    configure_ann,
    exact_top_k,
    recall_at_k,
)
from graph_bench.reporting import ResultCollector
from graph_bench.runner import BenchmarkOrchestrator, OrchestratorConfig
from graph_bench.types import ScaleConfig, Status

from .dict_adapter import DictAdapter

SCALE = ScaleConfig(name="tiny", nodes=1000, edges=10, warmup_iterations=1, measurement_iterations=2)


@pytest.fixture(autouse=True)
def default_ann() -> Iterator[None]:
    """Every test starts from, and leaves, the default ANN settings."""
    configure_ann(**vars(AnnSettings()))
    yield
    configure_ann(**vars(AnnSettings()))


@pytest.fixture
def grafeo() -> Iterator[GrafeoAdapter]:
    adapter = GrafeoAdapter()
    adapter.connect(path=":memory:")
    yield adapter
    adapter.disconnect()


# ── Ground truth and recall ─────────────────────────────────────


def test_exact_top_k_ranks_by_cosine() -> None:
    ids = ["a", "b", "c", "d"]
    vectors = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]]
    assert exact_top_k(ids, vectors, [[1.0, 0.1], [0.0, 2.0]], 2) == [["a", "b"], ["c", "b"]]
    assert exact_top_k([], [], [[1.0, 0.0]], 2) == [[]]


def test_recall_at_k() -> None:
    truth = [["a", "b"], ["c", "d"]]
    assert recall_at_k([["a", "b"], ["c", "d"]], truth, 2) == 1.0
    assert recall_at_k([["b", "x"], ["y", "z"]], truth, 2) == 0.25
    assert recall_at_k([["a"], []], truth, 2) == 0.25  # short results count as misses


@pytest.mark.parametrize(
    "changes",
    [{"quantization": "lossy"}, {"delete_fraction": 1.0}, {"ef": 0}, {"sweep": (10, 0)}],
)
def test_invalid_ann_settings_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        configure_ann(**changes)


def test_configure_ann_applies_to_new_benchmarks() -> None:
    configure_ann(ef=99, quantization="scalar")
    assert ann_settings().ef == 99
    assert ann_settings().index_metrics()["quantization"] == "scalar"


# ── Registry ────────────────────────────────────────────────────


def test_old_vector_names_select_the_scan_benchmarks() -> None:
    for old, new in [
        ("vector_insert", "vector_scan_insert"),
        ("vector_knn", "vector_scan_knn"),
        ("vector_batch_search", "vector_scan_batch_search"),
        ("vector_recall", "vector_scan_recall"),
    ]:
        assert BenchmarkRegistry.get(old) is BenchmarkRegistry.get(new)
        assert old not in BenchmarkRegistry.list()
    assert BenchmarkRegistry.by_category("vector_ann") == ["vector_ann_build", "vector_ann_knn", "vector_ann_delete"]


# ── Grafeo ANN benchmarks ───────────────────────────────────────


def test_ann_knn_reports_recall_and_parameters(grafeo: GrafeoAdapter) -> None:
    configure_ann(m=12, ef_construction=64, ef=48)
    metrics = VectorAnnKnnBenchmark().run(grafeo, SCALE)
    custom = metrics.custom
    assert custom["implementation"] == "native"
    assert custom["recall_at_k"] >= 0.9
    assert (custom["k"], custom["m"], custom["ef_construction"], custom["ef"]) == (10, 12, 64, 48)
    assert custom["queries"] == 100
    assert metrics.items_processed == 2 * 100 * 10
    assert "sweep" not in custom


def test_ann_sweep_gives_a_recall_curve(grafeo: GrafeoAdapter) -> None:
    configure_ann(sweep=(10, 40, 160))
    sweep = VectorAnnKnnBenchmark().run(grafeo, SCALE).custom["sweep"]
    assert [point["ef"] for point in sweep] == [10, 40, 160]
    assert all(0.0 <= point["recall_at_k"] <= 1.0 and point["mean_query_ms"] > 0 for point in sweep)
    assert sweep[-1]["recall_at_k"] >= sweep[0]["recall_at_k"]


@pytest.mark.parametrize("quantization", ["scalar", "binary", "product"])
def test_ann_with_quantization(grafeo: GrafeoAdapter, quantization: str) -> None:
    configure_ann(quantization=quantization, queries=20)
    custom = VectorAnnKnnBenchmark().run(grafeo, SCALE).custom
    assert custom["quantization"] == quantization
    assert 0.0 <= custom["recall_at_k"] <= 1.0


def test_ann_delete_reports_the_recall_drop(grafeo: GrafeoAdapter) -> None:
    configure_ann(delete_fraction=0.3)
    custom = VectorAnnDeleteBenchmark().run(grafeo, SCALE).custom
    assert custom["deleted"] == 300
    assert grafeo.count_nodes(label="VectorNode") == 700
    assert custom["recall_drop"] == pytest.approx(custom["recall_before_delete"] - custom["recall_at_k"])
    assert custom["deleted_returned"] >= 0
    assert custom["short_results"] >= 0


def test_ann_delete_survives_an_index_that_loses_vectors(grafeo: GrafeoAdapter, monkeypatch: Any) -> None:
    # Stands in for the grafeo 0.5.44 bug where a delete leaves other vectors
    # unreachable: the benchmark must report the drop, not fail or hide it.
    bench = VectorAnnDeleteBenchmark()
    real_search = GrafeoAdapter.vector_ann_search
    lost = threading.Event()

    def lossy_search(self: GrafeoAdapter, query_vector: list[float], **kwargs: Any) -> list[tuple[str, float]]:
        hits = real_search(self, query_vector, **kwargs)
        return hits[: len(hits) // 2] if lost.is_set() else hits

    real_delete = GrafeoAdapter.delete_nodes

    def delete_then_lose(self: GrafeoAdapter, node_ids: Any) -> int:
        lost.set()
        return real_delete(self, node_ids)

    monkeypatch.setattr(GrafeoAdapter, "vector_ann_search", lossy_search)
    monkeypatch.setattr(GrafeoAdapter, "delete_nodes", delete_then_lose)
    custom = bench.run(grafeo, SCALE).custom
    assert custom["recall_at_k"] == pytest.approx(0.5, abs=0.05)
    assert custom["recall_drop"] > 0.4
    assert custom["short_results"] == 100


def test_vectors_are_indexed_when_loaded_through_insert_nodes(grafeo: GrafeoAdapter) -> None:
    # A list literal in GQL text stays out of grafeo's vector index; insert_nodes
    # uses the direct API, so every vector must be found.
    nodes = [{"id": f"v{i}", "embedding": [1.0, float(i)]} for i in range(50)]
    grafeo.insert_nodes(nodes, label="VectorNode")
    grafeo.create_vector_index("VectorNode", "embedding", dimensions=2)
    assert len(grafeo.vector_ann_search([1.0, 0.0], k=50, ef=200)) == 50


# ── Unsupported, never a fallback ───────────────────────────────


def test_ann_without_native_support_is_skipped_not_faked() -> None:
    adapter = DictAdapter()
    orchestrator = BenchmarkOrchestrator(config=OrchestratorConfig(scale=SCALE))
    collector = ResultCollector()
    for name in BenchmarkRegistry.by_category("vector_ann"):
        bench_cls = BenchmarkRegistry.get(name)
        assert bench_cls is not None
        collector.add_results(orchestrator.run([adapter], benchmarks=[bench_cls()], scale=SCALE).results)
    assert [r.status for r in collector.results] == [Status.SKIPPED] * 3
    assert all("unsupported: dict has no native vector index" in (r.error or "") for r in collector.results)
    assert all(r.metrics is None for r in collector.results)


def test_scan_benchmarks_run_on_every_adapter() -> None:
    from graph_bench.benchmarks.vector import VectorScanInsertBenchmark

    # The dict adapter has no vector index and still runs the scan family.
    metrics = VectorScanInsertBenchmark().run(DictAdapter(), SCALE)
    assert metrics.items_processed == 2 * 1000
