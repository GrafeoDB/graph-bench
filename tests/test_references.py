import csv
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from graph_bench.datasets import cache
from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable
from graph_bench.datasets.snb import SnbGraph
from graph_bench.suites import catalog, graphalytics, scans, snb


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
    pairs = sorted({(a, b) for a, b in edges} | {(b, a) for a, b in edges})
    with (root / "E.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["src", "dst"])
        w.writerows(pairs)
    node = NodeTable("V", (Column("id", "INT64"),), "V.csv")
    edge = EdgeTable("E", "V", "V", (), "E.csv")
    return Dataset("graphalytics", "tiny", root, (node,), (edge,), len(vertices), len(pairs))


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


def test_ic13_bindings_cover_unreachable_and_self(graph: SnbGraph) -> None:
    """LDBC IC13 returns -1 when no path exists and 0 when person1 = person2: the bindings hit both."""
    bindings = snb.bindings("snb_ic13", graph, 10)
    assert len(bindings) == 10
    answers = [snb.reference("snb_ic13", graph, b)[0][0] for b in bindings]
    assert -1 in answers
    assert 0 in answers
    assert any(a > 0 for a in answers)
    unreachable = bindings[answers.index(-1)]
    assert "person_1000" in (unreachable["person1Id"], unreachable["person2Id"])
    same = bindings[answers.index(0)]
    assert same["person1Id"] == same["person2Id"]


def _hand_graph() -> SnbGraph:
    """Five nodes: three comments (two created at the same time, one content with a comma and quotes) and two
    persons whose insertion order is not the aggregate's ORDER BY."""
    comments = [
        ("comment_1", 'Hi, "you"', 100),
        ("comment_10", "x", 300),
        ("comment_2", "it's", 300),
    ]
    persons = [("person_1", "Firefox", "male", 50, 500), ("person_2", "Chrome", "female", 40, 600)]
    return SnbGraph(
        nodes={
            "Comment": {c: {"id": c, "content": t, "creationDate": d, "length": len(t)} for c, t, d in comments},
            "Person": {
                p: {"id": p, "browserUsed": b, "gender": s, "birthday": born, "creationDate": d}
                for p, b, s, born, d in persons
            },
        },
        out={},
        incoming={},
    )


def test_scan_references_on_a_hand_graph() -> None:
    g = _hand_graph()
    assert scans.reference("scan_label", g, {}) == [["comment_1"], ["comment_10"], ["comment_2"]]
    # The window includes `from` and excludes `to`.
    assert scans.reference("scan_filter", g, {"from": 100, "to": 300}) == [["comment_1"]]
    assert scans.reference("scan_filter", g, {"from": 101, "to": 301}) == [["comment_10"], ["comment_2"]]
    assert scans.reference("scan_filter", g, {"from": 301, "to": 400}) == []
    assert scans.reference("scan_aggregate", g, {}) == [
        ["Chrome", "female", 1, 40, 600],
        ["Firefox", "male", 1, 50, 500],
    ]
    # Newest first; the tie at 300 is broken by id as a string ("comment_10" < "comment_2").
    assert scans.reference("scan_order_limit", g, {"minLength": 1}) == [
        ["comment_10", "x", 300],
        ["comment_2", "it's", 300],
        ["comment_1", 'Hi, "you"', 100],
    ]
    assert scans.reference("scan_order_limit", g, {"minLength": 4}) == [
        ["comment_2", "it's", 300],
        ["comment_1", 'Hi, "you"', 100],
    ]
    # Lengths 9, 1 and 4; the threshold is exclusive.
    assert scans.reference("scan_filter_avg", g, {"min": 0}) == [[3, 14 / 3]]
    assert scans.reference("scan_filter_avg", g, {"min": 1}) == [[2, 6.5]]
    assert scans.reference("scan_filter_avg", g, {"min": 4}) == [[1, 9]]


def test_scan_bindings_spread_over_the_comments(graph: SnbGraph) -> None:
    n = 10
    comments = graph.nodes["Comment"].values()
    assert scans.bindings("scan_label", graph, n) == [{}]
    assert scans.bindings("scan_aggregate", graph, n) == [{}]
    # Ten windows, each over about 10% of the comments by creationDate, moving forward.
    windows = scans.bindings("scan_filter", graph, n)
    assert len(windows) == n
    assert [w["from"] for w in windows] == sorted({w["from"] for w in windows})
    for w in windows:
        inside = sum(w["from"] <= c["creationDate"] < w["to"] for c in comments)
        assert 0.08 * len(comments) <= inside <= 0.12 * len(comments), w
    # Every distinct comment length (fewer than 10 on the generated data).
    lengths = sorted({c["length"] for c in comments})
    assert scans.bindings("scan_order_limit", graph, n) == [{"minLength": x} for x in lengths]
    assert all(
        len(scans.reference("scan_order_limit", graph, b)) == 20 for b in scans.bindings("scan_order_limit", graph, n)
    )
    # Ten distinct thresholds below the longest comment, so every binding counts at least one comment.
    thresholds = [b["min"] for b in scans.bindings("scan_filter_avg", graph, n)]
    assert len(thresholds) == n and thresholds == sorted(set(thresholds))
    assert thresholds[0] < lengths[0] and thresholds[-1] < lengths[-1]
    assert all(scans.reference("scan_filter_avg", graph, {"min": t})[0][0] > 0 for t in thresholds)


def test_scan_twins_share_the_references() -> None:
    by_id = {item.id: item for item in catalog.ITEMS}
    for qid in ("scan_label", "scan_filter", "scan_aggregate", "scan_order_limit", "scan_filter_avg"):
        base = by_id[qid]
        assert (base.category, base.model) == ("scans", "property_graph")
        for suffix, model, dialect, name in (
            ("sql", "sql_pgq", "sql_pgq", "SQL/PGQ"),
            ("rdf", "rdf", "sparql", "SPARQL"),
        ):
            twin = by_id[f"{qid}_{suffix}"]
            assert isinstance(twin, catalog.Query)
            assert (twin.category, twin.model, twin.same_as) == ("scans", model, qid)
            assert twin.title == f"{base.title} ({name})"
            assert twin.text[dialect].strip()
            assert catalog.reference_source(twin.id) is scans
    assert catalog.LOAD_QL_PROBES[("snb", "property_graph")] == "scan_order_limit"
    assert catalog.LOAD_QL_PROBES[("snb", "sql_pgq")] == "scan_order_limit_sql"


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
    assert len(ranks) == 1_001
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
