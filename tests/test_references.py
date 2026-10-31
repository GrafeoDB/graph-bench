import csv
import itertools
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import networkx
import pytest

from graph_bench.datasets import cache, multigraph
from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable
from graph_bench.datasets.snb import SnbGraph
from graph_bench.suites import catalog, graphalytics, hops, patterns, scans, snb


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


# Pattern matching ------------------------------------------------------------------------------------------


def _k4() -> SnbGraph:
    return _knows([("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")])


def _naive_cycles(g: SnbGraph, person: str, length: int) -> int:
    """The matches of the triangle (3) or square (4) text through `person`, enumerated as openCypher defines them:
    every sequence of `length` KNOWS edges in either direction, no edge twice (relationship isomorphism), leaving
    `person` and returning to it, kept by the text's WHERE (`b.id < c.id`; `b.id < d.id AND c.id <> $personId`)."""
    edges = [(src, dst) for src, out in g.out["KNOWS"].items() for dst, _ in out]
    found = 0
    for chosen in itertools.permutations(range(len(edges)), length):
        walk = [person]
        for e in chosen:
            src, dst = edges[e]
            if walk[-1] == src:
                walk.append(dst)
            elif walk[-1] == dst:
                walk.append(src)
            else:
                break
        if len(walk) != length + 1 or walk[-1] != person:
            continue
        if length == 3 and walk[1] < walk[2]:
            found += 1
        if length == 4 and walk[1] < walk[3] and walk[2] != person:
            found += 1
    return found


def test_triangle_reference_on_k4() -> None:
    k4 = _k4()
    assert patterns.reference("pattern_triangle", k4, {"personId": "a"}) == [[3]]
    assert patterns.reference("pattern_triangle_hub", k4, {"personId": "a"}) == [[3]]
    # Each of the four triangles once, whoever is `a`.
    assert patterns.reference("pattern_triangle_all", k4, {}) == [[4]]
    # A triangle with a tail: d is on no triangle; the whole graph has one.
    tail = _knows([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    assert patterns.reference("pattern_triangle", tail, {"personId": "a"}) == [[1]]
    assert patterns.reference("pattern_triangle", tail, {"personId": "d"}) == [[0]]
    assert patterns.reference("pattern_triangle_all", tail, {}) == [[1]]


def test_square_reference_counts_each_cycle_once() -> None:
    # K4: three 4-cycles through a vertex (a-b-c-d, a-b-d-c, a-c-b-d), each matched in two directions and kept
    # once by `b.id < d.id`.
    assert patterns.reference("pattern_square", _k4(), {"personId": "a"}) == [[3]]
    square = _knows([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    assert patterns.reference("pattern_square", square, {"personId": "a"}) == [[1]]
    # A triangle has no 4-cycle: the closed 4-edge walks through it repeat a vertex.
    triangle = _knows([("a", "b"), ("b", "c"), ("c", "a")])
    assert patterns.reference("pattern_square", triangle, {"personId": "a"}) == [[0]]


def test_cycle_references_match_naive_enumeration() -> None:
    sparse = networkx.gnm_random_graph(9, 17, seed=3)
    g = _knows([(f"p{u}", f"p{v}") for u, v in sparse.edges])
    squares = 0
    for person in sorted(g.nodes["Person"]):
        binding = {"personId": person}
        assert patterns.reference("pattern_triangle", g, binding) == [[_naive_cycles(g, person, 3)]], person
        assert patterns.reference("pattern_square", g, binding) == [[_naive_cycles(g, person, 4)]], person
        squares += patterns.reference("pattern_square", g, binding)[0][0]
    assert squares > 0
    total = sum(networkx.triangles(sparse).values()) // 3
    assert total > 0
    assert patterns.reference("pattern_triangle_all", g, {}) == [[total]]


def test_common_friends_reference() -> None:
    # a and e share b and c; d is a friend of a alone.
    g = _knows([("a", "b"), ("a", "c"), ("a", "d"), ("e", "b"), ("c", "e")])
    assert patterns.reference("pattern_common", g, {"person1Id": "a", "person2Id": "e"}) == [[2]]
    assert patterns.reference("pattern_common", g, {"person1Id": "b", "person2Id": "c"}) == [[2]]
    assert patterns.reference("pattern_common", g, {"person1Id": "a", "person2Id": "b"}) == [[0]]


def _join_graph() -> SnbGraph:
    """Two persons, three tags, three posts and a comment: p1 likes t1 and t2, p2 likes t2."""
    tags = {"t1": "Music", "t2": "Art", "t3": "Cooking"}
    nodes = {
        "Person": {"p1": {"id": "p1"}, "p2": {"id": "p2"}},
        "Tag": {t: {"id": t, "name": name} for t, name in tags.items()},
        "Post": {m: {"id": m} for m in ("m1", "m2", "m3")},
        "Comment": {"c1": {"id": "c1"}},
    }
    # m1 by p1 tagged t1 and t2; m2 by p2 tagged t1 and t2; m3 by p1 tagged t3; the comment by p1 tagged t1.
    out = {
        "HAS_INTEREST": {"p1": [("t1", {}), ("t2", {})], "p2": [("t2", {})]},
        "HAS_TAG": {
            "m1": [("t1", {}), ("t2", {})],
            "m2": [("t1", {}), ("t2", {})],
            "m3": [("t3", {})],
            "c1": [("t1", {})],
        },
        "HAS_CREATOR": {"m1": [("p1", {})], "m2": [("p2", {})], "m3": [("p1", {})], "c1": [("p1", {})]},
    }
    return SnbGraph(nodes=nodes, out=out, incoming={})


def test_join_reference_counts_posts_per_tag() -> None:
    # (p1, t1, m1), (p1, t2, m1), (p2, t2, m2): Art 2, Music 1. The comment is no Post, and p1 does not like t3.
    g = _join_graph()
    assert patterns.reference("pattern_join", g, {}) == [["Art", 2], ["Music", 1]]
    # A tie orders by tag: p2 likes t1 too, so (p2, t1, m2) makes Music 2.
    g.out["HAS_INTEREST"]["p2"].append(("t1", {}))
    assert patterns.reference("pattern_join", g, {}) == [["Art", 2], ["Music", 2]]


def _multi(root: Path, vertices: list[int], edges: list[tuple[int, int]]) -> Dataset:
    """A dataset with the graphalytics schema whose E rows are `edges` exactly: repeated rows and self-loops kept."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "V.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id"])
        w.writerows([v] for v in vertices)
    with (root / "E.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["src", "dst"])
        w.writerows(edges)
    node = NodeTable("V", (Column("id", "INT64"),), "V.csv")
    edge = EdgeTable("E", "V", "V", (), "E.csv")
    return Dataset("graphalytics", "tiny", root, (node,), (edge,), len(vertices), len(edges))


def _naive_two_hop(edges: list[tuple[int, int]]) -> dict[int, int]:
    """`MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V)` enumerated: every ordered pair of two different edges (by position, so
    a repeated row is a different edge) where the first ends where the second starts; matches per source `a`."""
    per_source: dict[int, int] = defaultdict(int)
    for i, (a, b) in enumerate(edges):
        for j, (b2, _) in enumerate(edges):
            if i != j and b == b2:
                per_source[a] += 1
    return dict(per_source)


def _naive_walks(edges: list[tuple[int, int]]) -> int:
    """The same pairs with an edge allowed twice (homomorphism)."""
    return sum(1 for _, b in edges for b2, _ in edges if b == b2)


def test_two_hop_reference_matches_naive_enumeration(tmp_path: Path) -> None:
    """A1 on the graphalytics form (each edge stored in both directions, no self-loop): every walk a->b->c is a
    trail, a->b->a included (two stored edges), so the count is the sum of the squared degrees."""
    undirected = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (2, 6)]
    edges = sorted({(a, b) for a, b in undirected} | {(b, a) for a, b in undirected})
    ds = _multi(tmp_path / "six", [1, 2, 3, 4, 5, 6], edges)
    naive = _naive_two_hop(edges)
    degree = {v: sum(1 for a, _ in edges if a == v) for v in range(1, 7)}
    assert sum(naive.values()) == sum(d * d for d in degree.values()) == _naive_walks(edges)
    assert patterns.reference("pm_two_hop_count", ds, {}) == [[sum(naive.values())]]
    # A2: per source below the cut, the largest counts, ties by id.
    top = sorted(naive.items(), key=lambda kv: (-kv[1], kv[0]))
    assert patterns.reference("pm_two_hop_top10", ds, {"cut": 7}) == [[a, n] for a, n in top]
    assert patterns.reference("pm_two_hop_top10", ds, {"cut": 3}) == [[a, n] for a, n in top if a < 3]
    assert patterns.reference("pm_two_hop_top10", ds, {"cut": 1}) == []
    # A3: out-degree, ties by id.
    expected = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))
    assert patterns.reference("agg_out_degree_top10", ds, {}) == [[v, d] for v, d in expected]
    # LIMIT 10 on a graph with more sources.
    star = [(0, v) for v in range(1, 13)] + [(v, 0) for v in range(1, 13)]
    big = _multi(tmp_path / "star", list(range(13)), star)
    assert len(patterns.reference("pm_two_hop_top10", big, {"cut": 13})) == 10
    assert patterns.reference("agg_out_degree_top10", big, {})[:2] == [[0, 12], [1, 1]]
    assert len(patterns.reference("agg_out_degree_top10", big, {})) == 10


def test_trail_reference_counts_each_edge_once(tmp_path: Path) -> None:
    """C1: openCypher never matches one relationship twice in a pattern, so a self-loop e gives no path (e, e);
    parallel edges are different relationships."""
    cases = {
        "loop": ([(1, 1)], 0, 1),
        "two loops": ([(1, 1), (1, 1)], 2, 4),
        "parallel": ([(1, 2), (1, 2), (2, 3)], 2, 2),
        "loop in the middle": ([(1, 2), (2, 2), (2, 3)], 3, 4),
    }
    for name, (edges, trails, walks) in cases.items():
        ds = _multi(tmp_path / name.replace(" ", "_"), [1, 2, 3], edges)
        assert sum(_naive_two_hop(edges).values()) == trails, name
        assert _naive_walks(edges) == walks, name
        assert patterns.reference("sem_trail_self_loop", ds, {}) == [[trails]], name
        assert patterns.reference("pm_two_hop_count", ds, {}) == [[trails]], name


def test_multigraph_fixture_has_loops_and_parallel_edges(tmp_path: Path) -> None:
    ds = multigraph.generate("small", tmp_path / "multigraph")
    assert (ds.name, ds.scale) == ("multigraph", "small")
    assert Dataset.load(ds.root) == ds
    [vertices] = ds.nodes
    [edge] = ds.edges
    assert (vertices.label, edge.type, edge.src, edge.dst) == ("V", "E", "V", "V")
    ids = [row["id"] for row in ds.rows(vertices)]
    edges = [(row["src"], row["dst"]) for row in ds.rows(edge)]
    assert (ds.node_count, ds.edge_count) == (len(ids), len(edges))
    assert any(a == b for a, b in edges), "a self-loop"
    assert len(set(edges)) < len(edges), "a multi-edge"
    assert any(edges.count((a, a)) > 1 for a, _ in edges), "parallel self-loops"
    assert any((b, a) in edges for a, b in edges if a != b), "an antiparallel pair"
    assert set(ids) - {v for e in edges for v in e}, "an isolated vertex"
    trails = sum(_naive_two_hop(edges).values())
    assert patterns.reference("sem_trail_self_loop", ds, {}) == [[trails]]
    assert trails < _naive_walks(edges)  # an engine that matches walks disagrees
    assert patterns.bindings("sem_trail_self_loop", ds, 10) == [{}]


def test_pattern_bindings(graph: SnbGraph, prepared: dict[tuple[str, str], Dataset]) -> None:
    persons = sorted(p for p in graph.nodes["Person"] if graph.out["KNOWS"].get(p) or graph.incoming["KNOWS"].get(p))
    degree = {p: len(graph.out["KNOWS"].get(p, [])) + len(graph.incoming["KNOWS"].get(p, [])) for p in persons}
    median = sorted(degree.values())[len(degree) // 2]
    for qid in ("pattern_triangle", "pattern_square"):
        ids = [b["personId"] for b in patterns.bindings(qid, graph, 10)]
        assert len(ids) == 10 and ids == sorted(set(ids)), qid
        assert all(0 < degree[p] <= median for p in ids), qid
    # The hubs: the top 1% of the persons by degree (10 of 1,001 on snb-small), all of them here.
    hubs = [b["personId"] for b in patterns.bindings("pattern_triangle_hub", graph, 10)]
    top = sorted(degree, key=lambda p: (-degree[p], p))[: len(graph.nodes["Person"]) // 100]
    assert hubs == sorted(top) and len(hubs) == 10
    assert min(degree[p] for p in hubs) > 10 * median
    fewer = [b["personId"] for b in patterns.bindings("pattern_triangle_hub", graph, 4)]
    assert len(fewer) == 4 and set(fewer) <= set(top)
    # Common friends: pairs of persons with friends half the id-sorted list apart, as IC13.
    pairs = patterns.bindings("pattern_common", graph, 10)
    half = len(persons) // 2
    assert len(pairs) == 10
    assert len({frozenset(b.values()) for b in pairs}) == 10  # no pair twice, in either order
    for b in pairs:
        i = persons.index(b["person1Id"])
        assert i < half and b["person2Id"] == persons[i + half]
    for qid in ("pattern_join", "pattern_triangle_all"):
        assert patterns.bindings(qid, graph, 10) == [{}]
    ga = prepared[("graphalytics", "small")]
    for qid in ("pm_two_hop_count", "agg_out_degree_top10"):
        assert patterns.bindings(qid, ga, 10) == [{}]
    # A2: ten cuts, each taking the next tenth of the vertex ids, the last every vertex.
    cuts = [b["cut"] for b in patterns.bindings("pm_two_hop_top10", ga, 10)]
    vertex_ids = sorted(row["id"] for row in ga.rows(ga.nodes[0]))
    assert len(cuts) == 10 and cuts == sorted(set(cuts))
    assert cuts[0] > vertex_ids[0] and cuts[-1] > vertex_ids[-1]
    assert all(len(patterns.reference("pm_two_hop_top10", ga, {"cut": c})) == 10 for c in cuts)


def test_pattern_references_on_generated_data(graph: SnbGraph, prepared: dict[tuple[str, str], Dataset]) -> None:
    """Every case has nonzero answers on snb-small, and A1 is the sum of the squared degrees on graphalytics."""
    for qid in ("pattern_triangle", "pattern_square", "pattern_triangle_hub", "pattern_common"):
        answers = [patterns.reference(qid, graph, b)[0][0] for b in patterns.bindings(qid, graph, 10)]
        assert any(a > 0 for a in answers), qid
    join = patterns.reference("pattern_join", graph, {})
    assert join and [r[1] for r in join] == sorted((r[1] for r in join), reverse=True)
    assert patterns.reference("pattern_triangle_all", graph, {})[0][0] > 0
    ga = prepared[("graphalytics", "small")]
    [[paths]] = patterns.reference("pm_two_hop_count", ga, {})
    [edge] = ga.edges
    degree: dict[int, int] = defaultdict(int)
    for row in ga.rows(edge):
        degree[row["src"]] += 1
    assert paths == sum(d * d for d in degree.values())
    top = patterns.reference("agg_out_degree_top10", ga, {})
    assert top[0][1] == max(degree.values()) and len(top) == 10


_PATTERN_CORE = ("pattern_triangle", "pattern_square", "pattern_common", "pattern_join")


def test_pattern_items_and_twins() -> None:
    by_id = {item.id: item for item in catalog.ITEMS}
    expected = {
        "pattern_triangle": ("Triangles through a person", "snb", False),
        "pattern_square": ("4-cycles through a person", "snb", False),
        "pattern_common": ("Common friends", "snb", False),
        "pattern_join": ("Multi-pattern join", "snb", True),
        "pattern_triangle_hub": ("Triangles through a hub person", "snb", False),
        "pattern_triangle_all": ("Triangles, whole graph", "snb", False),
        "pm_two_hop_count": ("2-hop paths, whole graph", "graphalytics", False),
        "pm_two_hop_top10": ("2-hop paths per source, top 10", "graphalytics", True),
        "agg_out_degree_top10": ("Out-degree, top 10", "graphalytics", True),
        "sem_trail_self_loop": ("Trail semantics with self-loops", "multigraph", False),
    }
    assert tuple(expected) == patterns.PATTERN_IDS
    for qid, (title, dataset, ordered) in expected.items():
        item = by_id[qid]
        assert isinstance(item, catalog.Query)
        assert (item.title, item.category, item.model, item.dataset) == (
            title,
            "pattern_matching",
            "property_graph",
            dataset,
        )
        assert (item.ordered, item.rel_eps, item.same_as) == (ordered, 0.0, "")
        assert catalog.reference_source(qid) is patterns
        assert item.text == {"cypher": item.text["cypher"], "helix": qid}, qid
    names = {"sql": ("sql_pgq", "sql_pgq", "SQL/PGQ"), "rdf": ("rdf", "sparql", "SPARQL")}
    for qid in _PATTERN_CORE:
        base = by_id[qid]
        for suffix, (model, dialect, name) in names.items():
            twin = by_id[f"{qid}_{suffix}"]
            assert isinstance(twin, catalog.Query)
            assert (twin.category, twin.model, twin.same_as, twin.dataset) == ("pattern_matching", model, qid, "snb")
            assert (twin.title, twin.ordered, twin.rel_eps) == (f"{base.title} ({name})", base.ordered, 0.0)
            assert set(twin.text) == {dialect}
            assert catalog.reference_source(twin.id) is patterns
    # Twins for the four core items only.
    others = set(expected) - set(_PATTERN_CORE)
    assert not {f"{q}_{s}" for q in others for s in names} & set(by_id)
    # C1 runs the A1 text in every dialect; the hub case runs the triangle text.
    c1, a1 = by_id["sem_trail_self_loop"], by_id["pm_two_hop_count"]
    hub, triangle = by_id["pattern_triangle_hub"], by_id["pattern_triangle"]
    assert isinstance(c1, catalog.Query) and isinstance(a1, catalog.Query)
    assert isinstance(hub, catalog.Query) and isinstance(triangle, catalog.Query)
    assert c1.text["cypher"] == a1.text["cypher"]
    assert hub.text["cypher"] == triangle.text["cypher"]


def test_analytics_texts_are_the_frozen_ones() -> None:
    """A1 to A3 exactly as in the analytics brief (.claude/todo/grafeo-0.6.1-analytics.md)."""
    by_id = {item.id: item for item in catalog.ITEMS}
    frozen = {
        "pm_two_hop_count": "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN count(*) AS paths",
        "pm_two_hop_top10": (
            "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) WHERE a.id < $cut RETURN a.id AS id, count(*) AS paths "
            "ORDER BY paths DESC, id ASC LIMIT 10"
        ),
        "agg_out_degree_top10": (
            "MATCH (a:V)-[:E]->(:V) RETURN a.id AS id, count(*) AS degree ORDER BY degree DESC, id ASC LIMIT 10"
        ),
    }
    for qid, text in frozen.items():
        item = by_id[qid]
        assert isinstance(item, catalog.Query)
        assert item.text["cypher"] == text, qid
    brief = Path(__file__).resolve().parent.parent / ".claude" / "todo" / "grafeo-0.6.1-analytics.md"
    if brief.exists():
        content = brief.read_text(encoding="utf-8")
        for text in frozen.values():
            assert text in content


def test_pattern_texts_keep_the_distinctness_conditions() -> None:
    """Every text of the cyclic patterns keeps the conditions that make every matched node distinct, so relationship
    isomorphism and homomorphism give the same answer (see `patterns.py`)."""
    by_id = {item.id: item for item in catalog.ITEMS}
    conditions = {
        "pattern_triangle": {"cypher": ["b.id < c.id"], "sql_pgq": ["b.id < c.id"], "sparql": ["?bid < ?cid"]},
        "pattern_square": {
            "cypher": ["b.id < d.id", "c.id <> $personId"],
            "sql_pgq": ["b.id < d.id", "c.id <> $personId"],
            "sparql": ["?bid < ?did", "?cid != ?personId"],
        },
        "pattern_triangle_hub": {"cypher": ["b.id < c.id"]},
        "pattern_triangle_all": {"cypher": ["a.id < b.id AND b.id < c.id"]},
    }
    for qid, by_dialect in conditions.items():
        texts = {}
        for item_id in (qid, f"{qid}_sql", f"{qid}_rdf"):
            item = by_id.get(item_id)
            if isinstance(item, catalog.Query):
                texts.update({dialect: text for dialect, text in item.text.items() if dialect != "helix"})
        assert set(texts) == set(by_dialect), qid
        for dialect, needles in by_dialect.items():
            for needle in needles:
                assert needle in texts[dialect], (qid, dialect, needle)
