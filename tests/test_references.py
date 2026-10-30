import csv
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from graph_bench.datasets import cache
from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable
from graph_bench.datasets.snb import SnbGraph
from graph_bench.suites import catalog, graphalytics, snb


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[tuple[str, str], Dataset]]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAPH_BENCH_DATA_DIR", str(tmp_path_factory.mktemp("data")))
        yield cache.prepare("push")


@pytest.fixture(scope="module")
def base(prepared: dict[tuple[str, str], Dataset]) -> Path:
    """The selection's cache directory, which holds one directory per (dataset, scale)."""
    return prepared[("snb", "small")].root.parent


@pytest.fixture(scope="module")
def graph(prepared: dict[tuple[str, str], Dataset]) -> SnbGraph:
    return SnbGraph.from_dataset(prepared[("snb", "small")])


def _tiny(root: Path, vertices: list[int], edges: list[tuple[int, int]]) -> Dataset:
    """A hand-built Graphalytics dataset; edges are stored in both directions."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "V.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id"])
        w.writerows([v] for v in vertices)
    with (root / "E.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["src", "dst"])
        w.writerows(sorted({(a, b) for a, b in edges} | {(b, a) for a, b in edges}))
    node = NodeTable("V", (Column("id", "INT64"),), "V.csv")
    edge = EdgeTable("E", "V", "V", (), "E.csv")
    return Dataset("graphalytics", "tiny", root, (node,), (edge,))


def test_reference_is1_matches_props(graph: SnbGraph) -> None:
    binding = snb.bindings("snb_is1", graph, 10)[3]
    person = graph.nodes["Person"][binding["personId"]]
    city = graph.out["LIVES_IN"][binding["personId"]][0][0]
    expected = [
        person["firstName"],
        person["lastName"],
        person["birthday"],
        person["locationIP"],
        person["browserUsed"],
        city,
        person["gender"],
        person["creationDate"],
    ]
    assert snb.reference("snb_is1", graph, binding) == [expected]


def test_reference_ic13_self_and_unreachable(graph: SnbGraph) -> None:
    a = snb.bindings("snb_ic13", graph, 10)[0]["person1Id"]
    assert snb.reference("snb_ic13", graph, {"person1Id": a, "person2Id": a}) == [[0]]
    lonely = {"id": "person_lonely", "firstName": "Ada", "lastName": "Lovelace"}
    extended = SnbGraph(
        nodes={**graph.nodes, "Person": {**graph.nodes["Person"], lonely["id"]: lonely}},
        out=graph.out,
        incoming=graph.incoming,
    )
    assert snb.reference("snb_ic13", extended, {"person1Id": a, "person2Id": lonely["id"]}) == [[-1]]
    assert snb.reference("snb_ic13", extended, {"person1Id": lonely["id"], "person2Id": a}) == [[-1]]
    friend = graph.out["KNOWS"][a][0][0] if a in graph.out["KNOWS"] else graph.incoming["KNOWS"][a][0][0]
    assert snb.reference("snb_ic13", graph, {"person1Id": a, "person2Id": friend}) == [[1]]


def test_bfs_reference_depths(tmp_path: Path) -> None:
    ds = _tiny(tmp_path, [1, 2, 3, 4], [(1, 2), (2, 3)])
    assert graphalytics.bindings("graphalytics_bfs", ds) == [{"source": 2}]
    assert graphalytics.reference("graphalytics_bfs", ds, {"source": 1}) == {1: 0, 2: 1, 3: 2}
    assert graphalytics.reference("graphalytics_bfs", ds, {"source": 4}) == {4: 0}


def test_wcc_reference_min_member(tmp_path: Path) -> None:
    ds = _tiny(tmp_path, [3, 4, 5, 7, 9], [(7, 3), (4, 7), (9, 5)])
    assert graphalytics.bindings("graphalytics_wcc", ds) == [{}]
    assert graphalytics.reference("graphalytics_wcc", ds, {}) == {3: 3, 4: 3, 7: 3, 5: 5, 9: 5}
    assert graphalytics.min_member({3: "x", 4: "x", 7: "x", 5: 0, 9: 0}) == {3: 3, 4: 3, 7: 3, 5: 5, 9: 5}


def test_pr_reference_sums_to_one(prepared: dict[tuple[str, str], Dataset]) -> None:
    ranks = graphalytics.reference("graphalytics_pr", prepared[("graphalytics", "small")], {})
    assert len(ranks) == 1_000
    assert sum(ranks.values()) == pytest.approx(1.0, abs=1e-9)


def test_prepare_writes_references(prepared: dict[tuple[str, str], Dataset], base: Path) -> None:
    path = prepared[("snb", "small")].root / "references" / "snb_is1-small.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert len(doc["bindings"]) == 10
    assert len(doc["answers"]) == 10
    is1 = next(i for i in catalog.ITEMS if i.id == "snb_is1")
    assert catalog.reference_path(is1, "small", base=base) == path
    for item in catalog.ITEMS:
        if item.category in catalog.SELECTIONS["push"].categories:
            assert catalog.reference_path(item, "small", base=base).exists(), item.id


def test_load_reference_restores_int_keys(base: Path) -> None:
    pr = next(i for i in catalog.ITEMS if i.id == "graphalytics_pr")
    bindings, answers = catalog.load_reference(catalog.reference_path(pr, "small", base=base))
    assert bindings == [{}]
    assert len(answers) == 1 and isinstance(answers[0], dict)
    assert all(isinstance(k, int) for k in answers[0])
    bfs = next(i for i in catalog.ITEMS if i.id == "graphalytics_bfs")
    bfs_bindings, bfs_answers = catalog.load_reference(catalog.reference_path(bfs, "small", base=base))
    assert bfs_answers[0][bfs_bindings[0]["source"]] == 0
    assert (base / "graphalytics-small" / "references" / "graphalytics_bfs-small.json").exists()
