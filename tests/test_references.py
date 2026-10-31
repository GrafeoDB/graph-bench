import csv
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import networkx
import pytest

from graph_bench.datasets import cache
from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable
from graph_bench.datasets.snb import SnbGraph
from graph_bench.suites import catalog, graphalytics, hops, scans, snb


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


def _knows(pairs: list[tuple[str, str]]) -> SnbGraph:
    """Persons and the KNOWS edges `pairs` (stored as directed edges, src to dst, as the generator does)."""
    out: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for src, dst in pairs:
        out[src].append((dst, {"creationDate": 0}))
        incoming[dst].append((src, {"creationDate": 0}))
    persons = sorted({p for pair in pairs for p in pair})
    return SnbGraph(
        nodes={"Person": {p: {"id": p} for p in persons}},
        out={"KNOWS": dict(out)},
        incoming={"KNOWS": dict(incoming)},
    )


def _hops(g: SnbGraph, person: str) -> dict[str, int]:
    return {qid: hops.reference(qid, g, {"personId": person})[0][0] for qid in hops.HOP_IDS}


def test_hop_references_exclude_repeated_edges() -> None:
    # Path a-b: the only 2-edge walk from a is a-b-a, which repeats the edge and returns to a, so hop_2 is 0.
    edge = _knows([("a", "b")])
    assert _hops(edge, "a") == {"hop_1": 1, "hop_2": 0, "hop_var": 1}
    # Path a-b-c: a-b-c reaches c; a-b-a does not count.
    path = _knows([("a", "b"), ("c", "b")])
    assert _hops(path, "a") == {"hop_1": 1, "hop_2": 1, "hop_var": 2}
    # Triangle a-b-c: the 3-edge trail a-b-c-a ends at a itself, which `f.id <> $personId` removes from hop_var;
    # the two 2-edge walks a-b-c and a-c-b reach c and b.
    triangle = _knows([("a", "b"), ("b", "c"), ("c", "a")])
    assert _hops(triangle, "a") == {"hop_1": 2, "hop_2": 2, "hop_var": 2}
    # Path a-b-c-d-e: hop_var stops at d, 3 hops away; edge direction does not matter.
    long_path = _knows([("a", "b"), ("c", "b"), ("c", "d"), ("d", "e")])
    assert _hops(long_path, "a") == {"hop_1": 1, "hop_2": 1, "hop_var": 3}


def test_hop_references_count_distinct_persons_not_walks() -> None:
    # a knows b, c and d, each of whom knows e: three 2-edge walks end at e, one person. hop_var reaches b, c, d and
    # e by many walks (a-b-e-c, a-c-e-b, ...), four persons.
    fan = _knows([("a", "b"), ("a", "c"), ("a", "d"), ("b", "e"), ("c", "e"), ("d", "e")])
    assert _hops(fan, "a") == {"hop_1": 3, "hop_2": 1, "hop_var": 4}
    # Square a-b-c-d-a: a-b-c and a-d-c end at the same person.
    square = _knows([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    assert _hops(square, "a") == {"hop_1": 2, "hop_2": 1, "hop_var": 3}


def test_hop_references_reject_repeated_pairs_and_self_loops() -> None:
    """The texts' node-distinctness conditions equal edge distinctness only on a simple graph: a generator change
    that breaks it must fail the references, not change their meaning."""
    with pytest.raises(ValueError, match="KNOWS pair"):
        hops.reference("hop_1", _knows([("a", "b"), ("b", "a")]), {"personId": "a"})
    with pytest.raises(ValueError, match="self-loop"):
        hops.reference("hop_2", _knows([("a", "b"), ("b", "b")]), {"personId": "a"})


def test_hop_var_equals_bfs_ball(graph: SnbGraph) -> None:
    """`*1..3` is the radius-3 ball around the person, without the person, in either match semantics."""

    def ball(g: SnbGraph, person: str) -> int:
        nx_graph = networkx.Graph()
        nx_graph.add_edges_from((src, dst) for src, edges in g.out["KNOWS"].items() for dst, _ in edges)
        return len(networkx.single_source_shortest_path_length(nx_graph, person, cutoff=3)) - 1

    bindings = hops.bindings("hop_var", graph, 10)
    assert len(bindings) == 10
    for b in bindings:
        assert hops.reference("hop_var", graph, b) == [[ball(graph, b["personId"])]], b
    # snb-small is dense enough that every ball above holds every person with friends; a sparse random graph has
    # balls of many sizes.
    sparse = networkx.gnm_random_graph(300, 330, seed=7)
    g = _knows([(f"p{u}", f"p{v}") for u, v in sparse.edges])
    sizes = {hops.reference("hop_var", g, {"personId": f"p{n}"})[0][0] for n in sparse if sparse.degree(n)}
    assert len(sizes) > 10
    for n in sparse:
        if sparse.degree(n):
            assert hops.reference("hop_var", g, {"personId": f"p{n}"}) == [[ball(g, f"p{n}")]], n


def test_hop_bindings_spread_over_persons_with_friends(graph: SnbGraph) -> None:
    persons = sorted(p for p in graph.nodes["Person"] if graph.out["KNOWS"].get(p) or graph.incoming["KNOWS"].get(p))
    degree = {p: len(graph.out["KNOWS"].get(p, [])) + len(graph.incoming["KNOWS"].get(p, [])) for p in persons}
    median = sorted(degree.values())[len(degree) // 2]
    for qid in hops.HOP_IDS:
        bindings = hops.bindings(qid, graph, 10)
        assert len(bindings) == 10
        ids = [b["personId"] for b in bindings]
        assert ids == sorted(set(ids)) and set(ids) <= set(persons)
        # A hub (more than ten times the median degree) is among them.
        assert any(degree[p] > 10 * median for p in ids), qid
        assert all(hops.reference(qid, graph, b)[0][0] > 0 for b in bindings)


def test_hop_twins_share_the_references() -> None:
    by_id = {item.id: item for item in catalog.ITEMS}
    twins = {
        "hop_1": ("sql", "rdf"),
        "hop_2": ("sql", "rdf"),
        "hop_var": ("sql",),  # SPARQL 1.1 has no bounded repetition of a property path
    }
    names = {"sql": ("sql_pgq", "sql_pgq", "SQL/PGQ"), "rdf": ("rdf", "sparql", "SPARQL")}
    for qid, suffixes in twins.items():
        base = by_id[qid]
        assert isinstance(base, catalog.Query)
        assert (base.category, base.model, base.ordered, base.rel_eps) == ("hops", "property_graph", False, 0.0)
        for suffix in suffixes:
            model, dialect, name = names[suffix]
            twin = by_id[f"{qid}_{suffix}"]
            assert isinstance(twin, catalog.Query)
            assert (twin.category, twin.model, twin.same_as, twin.rel_eps) == ("hops", model, qid, 0.0)
            assert twin.title == f"{base.title} ({name})"
            assert twin.text[dialect].strip()
            assert catalog.reference_source(twin.id) is hops
    assert "hop_var_rdf" not in by_id
    assert not {"hop_3", "hop_3_sql", "hop_3_rdf"} & set(by_id)
    # IC13 stays in the SNB module, in Hops.
    assert (by_id["snb_ic13"].category, catalog.reference_source("snb_ic13")) == ("hops", snb)


def test_hop_texts_keep_the_distinctness_conditions() -> None:
    """Every text of hop_2 and hop_var (each dialect, each twin) keeps the node-distinctness condition that makes
    relationship isomorphism and homomorphism give the same answer."""
    by_id = {item.id: item for item in catalog.ITEMS}
    conditions = {
        "hop_2": {
            "cypher": ["f.id <> $personId"],
            "sql_pgq": ["f.id <> $personId"],
            "sparql": ["?fid != ?personId"],
        },
        "hop_var": {"cypher": ["f.id <> $personId"], "sql_pgq": ["f.id <> $personId"]},
    }
    for qid, by_dialect in conditions.items():
        texts = {
            dialect: text
            for item_id in (qid, f"{qid}_sql", f"{qid}_rdf")
            if item_id in by_id
            for dialect, text in by_id[item_id].text.items()  # type: ignore[union-attr]
            if dialect != "helix"
        }
        assert set(texts) == set(by_dialect), qid
        for dialect, needles in by_dialect.items():
            for needle in needles:
                assert needle in texts[dialect], (qid, dialect, needle)


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
