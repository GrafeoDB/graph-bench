"""Pattern matching: cyclic patterns, common neighbours and a multi-pattern join on SNB, whole-graph 2-hop counts on
Graphalytics, and the trail-semantics check on the multigraph fixture.

Ten openCypher queries with deterministic bindings and Python reference answers, all compared exactly (counts):

- `pattern_triangle`, `pattern_square`, `pattern_common`, `pattern_join` (ported from the legacy `pattern.py`
  triangle count and common neighbours, plus the cycles and joins of the spec), each with an SQL/PGQ twin
  (`<id>_sql`) and a SPARQL twin (`<id>_rdf`) that share its bindings and references (`same_as`), so DuckDB +
  DuckPGQ and Oxigraph take part.
- `pattern_triangle_hub`: the triangle text from hub persons; `pattern_triangle_all`: every triangle of the KNOWS
  graph once. These measure the hub blow-up and worst-case-optimal joins; the median-degree cases above are the
  guard.
- `pm_two_hop_count` (A1), `pm_two_hop_top10` (A2), `agg_out_degree_top10` (A3): the analytics brief's texts
  (`.claude/todo/grafeo-0.6.1-analytics.md`), frozen after the baseline, on Graphalytics.
- `sem_trail_self_loop` (C1): the A1 text on the multigraph fixture (`datasets/multigraph.py`), whose self-loops
  make trails and walks differ. It is answer-only (`catalog.ANSWER_ONLY_DATASETS`): no budget and no ranking.

The hub, whole-graph and analytics items are openCypher only (DuckDB and Oxigraph show the model gate's ⊘).

Match semantics. openCypher matches under relationship isomorphism (no edge twice in one match); some engines match
walks (homomorphism). Every SNB text binds pairwise distinct persons, so both give the same answer on a graph with
unique KNOWS pairs and no self-loop (`hops._adjacency` checks both): there an edge is identified by its two
persons, and distinct persons along a cycle mean distinct edges.

- Triangle `a-b-c-a`: b and c are neighbours of a (so not a), adjacent to each other (so b is not c), and
  `b.id < c.id` keeps one of the two directions round the triangle.
- Square `a-b-c-d-a`: `c.id <> $personId` makes c not a, `b.id < d.id` makes b not d and keeps one of the two
  directions; adjacent persons differ. So the four persons are distinct and each 4-cycle through a counts once
  (chords allowed: the pattern is not induced).
- Common friends `a-c-b`: the two persons of a binding differ (`bindings`), so the two edges differ.
- Whole-graph triangles: `a.id < b.id AND b.id < c.id` orders the three persons, one match per triangle.
- The join's three edges have three types, so no edge can repeat, and its nodes have three labels.

The 2-hop texts have no such condition, by design: C1 checks the semantics itself. A match of
`(a)-[:E]->(b)-[:E]->(c)` is a pair of edges (e1, e2) with e1 ending where e2 starts. Through a vertex v there are
in(v) * out(v) such pairs, and a pair repeats an edge (e1 = e2) exactly when e1 is a self-loop at v, one pair per
self-loop. So the trail count is the sum of in(v) * out(v) minus the self-loops, and the walk count is the sum
alone. Graphalytics stores each KNOWS pair as two edges and has no self-loop, so there both are the sum of the
squared degrees (a->b->a is two different edges). Per source a (A2) it is the sum over the out-edges (a, b) of
out(b), minus one where b = a. The references compute these formulas; tests/test_references.py checks them against
a naive enumeration of edge pairs.

Bindings: triangles and squares use persons with friends whose KNOWS degree is at most the median degree, spread
evenly over the id-sorted list (bounds the hub enumeration). The hub case uses the top 1% of all persons by KNOWS
degree (ties by id), spread evenly over that group sorted by id. Common friends use pairs of persons with friends
half the id-sorted list apart, as IC13, the first person spread over the first half (so no pair repeats). A2 uses
`n` cuts: cut i is one above the vertex id at the end of the (i + 1)-th n-th of the id-sorted vertices, so every
binding has sources and the last takes every vertex. The other items have one empty binding.

The SQL/PGQ texts follow ruling R6 as `snb_sql.py` and `hops.py` do: `GRAPH_TABLE (snb MATCH ...)`, the graph
pattern WHERE after the pattern with the openCypher conditions, `$name` parameters, a variable on every edge pattern.
A cycle repeats its first element variable (`...-[k3:KNOWS]-(a:Person)`), which the standard reads as the same
node; the join is a comma-separated list of path patterns sharing variables. A repeated variable carries its label
again: the standard allows a label expression on every occurrence (each one constrains the same node, so the
meaning is the openCypher one), and DuckPGQ reads a bare `(a)` as the label `a` ("The label a is not registered in
property graph snb"), so the labelled form is the one more engines accept (R6). The count, distinct count,
grouping and order are the outer SQL query's. The SPARQL texts follow `snb_rdf.py` and `hops.py`: inline `VALUES`
blocks first, a named variable for every person of the pattern, each KNOWS step the property path
`(v:KNOWS|^v:KNOWS)`, the conditions as FILTERs over the id literals (compared as strings, as the openCypher ids).
SPARQL matches walks, which the conditions make equal to the openCypher answer.

HelixDB (dialect "helix") has no query text: its text is the name of a builder in
`graph_bench.databases.helix_queries`, or a reason the DSL cannot express the query.
"""

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from graph_bench.answers import canonical
from graph_bench.datasets.rdf import PREFIXES
from graph_bench.datasets.schema import Dataset
from graph_bench.datasets.snb import SnbGraph
from graph_bench.suites import Query
from graph_bench.suites.hops import _adjacency
from graph_bench.suites.snb import _spread, _with_friends

# The openCypher items, in category order.
PATTERN_IDS = (
    "pattern_triangle",
    "pattern_square",
    "pattern_common",
    "pattern_join",
    "pattern_triangle_hub",
    "pattern_triangle_all",
    "pm_two_hop_count",
    "pm_two_hop_top10",
    "agg_out_degree_top10",
    "sem_trail_self_loop",
)

# Counts: compared exactly.
_EXACT = 0.0
# Rows of the top-k analytics queries.
LIMIT = 10
# The hub persons: this share of all persons, by KNOWS degree.
HUB_SHARE = 100

_TRIANGLE = """\
MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a)
WHERE b.id < c.id
RETURN count(*) AS triangles"""

_SQUARE = """\
MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(d:Person)-[:KNOWS]-(a)
WHERE b.id < d.id AND c.id <> $personId
RETURN count(*) AS cycles"""

_COMMON = """\
MATCH (a:Person {id: $person1Id})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: $person2Id})
RETURN count(DISTINCT c.id) AS common"""

_JOIN = """\
MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(t), (m)-[:HAS_CREATOR]->(p)
RETURN t.name AS tag, count(*) AS posts
ORDER BY posts DESC, tag ASC"""

_TRIANGLE_ALL = """\
MATCH (a:Person)-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a)
WHERE a.id < b.id AND b.id < c.id
RETURN count(*) AS triangles"""

# The analytics brief's texts, character for character (frozen after the baseline).
_TWO_HOP_COUNT = "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN count(*) AS paths"
_TWO_HOP_TOP10 = (
    "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) WHERE a.id < $cut RETURN a.id AS id, count(*) AS paths "
    "ORDER BY paths DESC, id ASC LIMIT 10"
)
_OUT_DEGREE_TOP10 = "MATCH (a:V)-[:E]->(:V) RETURN a.id AS id, count(*) AS degree ORDER BY degree DESC, id ASC LIMIT 10"

_TRIANGLE_SQL = """\
SELECT count(*) AS triangles
FROM GRAPH_TABLE (snb
  MATCH (a:Person)-[k1:KNOWS]-(b:Person)-[k2:KNOWS]-(c:Person)-[k3:KNOWS]-(a:Person)
  WHERE a.id = $personId AND b.id < c.id
  COLUMNS (c.id AS cid)
)"""

_SQUARE_SQL = """\
SELECT count(*) AS cycles
FROM GRAPH_TABLE (snb
  MATCH (a:Person)-[k1:KNOWS]-(b:Person)-[k2:KNOWS]-(c:Person)-[k3:KNOWS]-(d:Person)-[k4:KNOWS]-(a:Person)
  WHERE a.id = $personId AND b.id < d.id AND c.id <> $personId
  COLUMNS (c.id AS cid)
)"""

_COMMON_SQL = """\
SELECT count(DISTINCT cid) AS common
FROM GRAPH_TABLE (snb
  MATCH (a:Person)-[k1:KNOWS]-(c:Person)-[k2:KNOWS]-(b:Person)
  WHERE a.id = $person1Id AND b.id = $person2Id
  COLUMNS (c.id AS cid)
)"""

_JOIN_SQL = """\
SELECT tag, count(*) AS posts
FROM GRAPH_TABLE (snb
  MATCH (p:Person)-[i:HAS_INTEREST]->(t:Tag), (m:Post)-[h:HAS_TAG]->(t:Tag), (m:Post)-[c:HAS_CREATOR]->(p:Person)
  COLUMNS (t.name AS tag)
)
GROUP BY tag
ORDER BY posts DESC, tag ASC"""

_TRIANGLE_RDF = f"""\
{PREFIXES}SELECT (COUNT(*) AS ?triangles) WHERE {{
  VALUES ?personId {{ $personId }}
  ?a a v:Person ; v:id ?personId ; (v:KNOWS|^v:KNOWS) ?b .
  ?b a v:Person ; v:id ?bid ; (v:KNOWS|^v:KNOWS) ?c .
  ?c a v:Person ; v:id ?cid ; (v:KNOWS|^v:KNOWS) ?a .
  FILTER(?bid < ?cid)
}}"""

_SQUARE_RDF = f"""\
{PREFIXES}SELECT (COUNT(*) AS ?cycles) WHERE {{
  VALUES ?personId {{ $personId }}
  ?a a v:Person ; v:id ?personId ; (v:KNOWS|^v:KNOWS) ?b .
  ?b a v:Person ; v:id ?bid ; (v:KNOWS|^v:KNOWS) ?c .
  ?c a v:Person ; v:id ?cid ; (v:KNOWS|^v:KNOWS) ?d .
  ?d a v:Person ; v:id ?did ; (v:KNOWS|^v:KNOWS) ?a .
  FILTER(?bid < ?did && ?cid != ?personId)
}}"""

_COMMON_RDF = f"""\
{PREFIXES}SELECT (COUNT(DISTINCT ?cid) AS ?common) WHERE {{
  VALUES ?person1Id {{ $person1Id }}
  VALUES ?person2Id {{ $person2Id }}
  ?a a v:Person ; v:id ?person1Id ; (v:KNOWS|^v:KNOWS) ?c .
  ?c a v:Person ; v:id ?cid ; (v:KNOWS|^v:KNOWS) ?b .
  ?b a v:Person ; v:id ?person2Id .
}}"""

_JOIN_RDF = f"""\
{PREFIXES}SELECT ?tag (COUNT(*) AS ?posts) WHERE {{
  ?p a v:Person ; v:HAS_INTEREST ?t .
  ?t a v:Tag ; v:name ?tag .
  ?m a v:Post ; v:HAS_TAG ?t ; v:HAS_CREATOR ?p .
}}
GROUP BY ?tag
ORDER BY DESC(?posts) ?tag"""

# (id, title, dataset, openCypher, ordered)
_ITEMS: tuple[tuple[str, str, str, str, bool], ...] = (
    ("pattern_triangle", "Triangles through a person", "snb", _TRIANGLE, False),
    ("pattern_square", "4-cycles through a person", "snb", _SQUARE, False),
    ("pattern_common", "Common friends", "snb", _COMMON, False),
    ("pattern_join", "Multi-pattern join", "snb", _JOIN, True),
    # The hub case runs the triangle text.
    ("pattern_triangle_hub", "Triangles through a hub person", "snb", _TRIANGLE, False),
    ("pattern_triangle_all", "Triangles, whole graph", "snb", _TRIANGLE_ALL, False),
    ("pm_two_hop_count", "2-hop paths, whole graph", "graphalytics", _TWO_HOP_COUNT, False),
    ("pm_two_hop_top10", "2-hop paths per source, top 10", "graphalytics", _TWO_HOP_TOP10, True),
    ("agg_out_degree_top10", "Out-degree, top 10", "graphalytics", _OUT_DEGREE_TOP10, True),
    # C1 runs the A1 text (HelixDB: the A1 builder).
    ("sem_trail_self_loop", "Trail semantics with self-loops", "multigraph", _TWO_HOP_COUNT, False),
)

# The twins of the four core items: (id, SQL/PGQ, SPARQL).
_TWINS: tuple[tuple[str, str, str], ...] = (
    ("pattern_triangle", _TRIANGLE_SQL, _TRIANGLE_RDF),
    ("pattern_square", _SQUARE_SQL, _SQUARE_RDF),
    ("pattern_common", _COMMON_SQL, _COMMON_RDF),
    ("pattern_join", _JOIN_SQL, _JOIN_RDF),
)

_BASE = {qid: (title, ordered) for qid, title, _, _, ordered in _ITEMS}

# The openCypher items, then the SQL/PGQ twins, then the SPARQL twins: the category's rows in that order.
QUERIES: tuple[Query, ...] = (
    *(
        Query(qid, title, "pattern_matching", "property_graph", dataset, {"cypher": text, "helix": qid}, o, _EXACT)
        for qid, title, dataset, text, o in _ITEMS
    ),
    *(
        Query(
            f"{qid}_sql",
            f"{_BASE[qid][0]} (SQL/PGQ)",
            "pattern_matching",
            "sql_pgq",
            "snb",
            {"sql_pgq": sql},
            _BASE[qid][1],
            _EXACT,
            qid,
        )
        for qid, sql, _ in _TWINS
    ),
    *(
        Query(
            f"{qid}_rdf",
            f"{_BASE[qid][0]} (SPARQL)",
            "pattern_matching",
            "rdf",
            "snb",
            {"sparql": sparql},
            _BASE[qid][1],
            _EXACT,
            qid,
        )
        for qid, _, sparql in _TWINS
    ),
)


# References: SNB --------------------------------------------------------------------------------------------

Neighbours = dict[str, set[str]]

# The neighbour sets of the last SNB graph asked for (held, so its identity cannot be reused).
_SNB_CACHE: tuple[SnbGraph, Neighbours] | None = None


def _neighbours(g: SnbGraph) -> Neighbours:
    """Person -> the persons it shares a KNOWS edge with, in either direction. `hops._adjacency` raises on a
    self-loop or a repeated pair, on which the texts' conditions would no longer mean distinct edges."""
    global _SNB_CACHE
    if _SNB_CACHE is not None and _SNB_CACHE[0] is g:
        return _SNB_CACHE[1]
    nb = {p: {f for f, _ in edges} for p, edges in _adjacency(g).items()}
    _SNB_CACHE = (g, nb)
    return nb


def _triangle(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    nb = _neighbours(g)
    a = b["personId"]
    na = nb.get(a, set())
    return [[sum(1 for x in na for y in nb[x] if y in na and x < y)]]


def _square(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    """Paths a-x-y-z-a with x < z and y not a; the other persons differ from their neighbours on the path."""
    nb = _neighbours(g)
    a = b["personId"]
    na = nb.get(a, set())
    count = 0
    for x in na:
        for y in nb[x]:
            if y != a:
                count += sum(1 for z in nb[y] & na if x < z)
    return [[count]]


def _common(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    nb = _neighbours(g)
    return [[len(nb.get(b["person1Id"], set()) & nb.get(b["person2Id"], set()))]]


def _triangle_all(g: SnbGraph, _: dict[str, Any]) -> list[list[Any]]:
    nb = _neighbours(g)
    count = 0
    for a, na in nb.items():
        for x in na:
            if a < x:
                count += sum(1 for y in nb[x] if x < y and y in na)
    return [[count]]


def _join(g: SnbGraph, _: dict[str, Any]) -> list[list[Any]]:
    """Per tag name, the (person, tag, post) matches: a post tagged with a tag its creator is interested in. Every
    edge is one match element, so a repeated edge would count again (the generated data has none)."""
    names = {t: row["name"] for t, row in g.nodes.get("Tag", {}).items()}
    persons, posts = g.nodes.get("Person", {}), g.nodes.get("Post", {})
    interests = g.out.get("HAS_INTEREST", {})
    tags = g.out.get("HAS_TAG", {})
    counts: Counter[str] = Counter()
    for m, creators in g.out.get("HAS_CREATOR", {}).items():
        if m not in posts:
            continue
        for p, _ in creators:
            if p not in persons:
                continue
            liked = Counter(t for t, _ in interests.get(p, []) if t in names)
            for t, _ in tags.get(m, []):
                counts[names.get(t, "")] += liked[t]
    rows = sorted(((tag, n) for tag, n in counts.items() if n), key=lambda row: (-row[1], row[0]))
    return [[tag, n] for tag, n in rows]


# References: Graphalytics and the multigraph fixture ------------------------------------------------------


@dataclass(frozen=True)
class _Edges:
    """The E edges between V vertices, each row one edge: out-neighbours with repeats, in- and out-degrees and the
    number of self-loops."""

    vertices: list[int]
    out: dict[int, list[int]]
    indegree: Counter[int]
    loops: int


_GRAPH_CACHE: tuple[Dataset, _Edges] | None = None


def _edges(ds: Dataset) -> _Edges:
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None and _GRAPH_CACHE[0] is ds:
        return _GRAPH_CACHE[1]
    vertex_table = next(t for t in ds.nodes if t.label == "V")
    edge_table = next(t for t in ds.edges if t.type == "E")
    vertices = sorted(row["id"] for row in ds.rows(vertex_table))
    out: dict[int, list[int]] = defaultdict(list)
    indegree: Counter[int] = Counter()
    loops = 0
    for row in ds.rows(edge_table):
        src, dst = row["src"], row["dst"]
        out[src].append(dst)
        indegree[dst] += 1
        loops += src == dst
    edges = _Edges(vertices, dict(out), indegree, loops)
    _GRAPH_CACHE = (ds, edges)
    return edges


def _paths_from(e: _Edges, a: int) -> int:
    """The 2-hop trails from `a`: for each out-edge (a, b), the out-edges of b, without the edge itself when it is a
    self-loop (b = a)."""
    return sum(len(e.out.get(b, [])) - (b == a) for b in e.out.get(a, []))


def _two_hop_count(ds: Dataset, _: dict[str, Any]) -> list[list[Any]]:
    """The trails a->b->c: the sum of in(v) * out(v) over the vertices, minus one per self-loop."""
    e = _edges(ds)
    walks = sum(e.indegree[v] * len(targets) for v, targets in e.out.items())
    return [[walks - e.loops]]


def _top(rows: list[tuple[int, int]]) -> list[list[Any]]:
    """Rows with a nonzero count (a group exists only with a match), largest first, ties by id, the first 10."""
    ranked = sorted(((v, n) for v, n in rows if n), key=lambda row: (-row[1], row[0]))
    return [[v, n] for v, n in ranked[:LIMIT]]


def _two_hop_top10(ds: Dataset, b: dict[str, Any]) -> list[list[Any]]:
    e = _edges(ds)
    return _top([(a, _paths_from(e, a)) for a in e.out if a < b["cut"]])


def _out_degree_top10(ds: Dataset, _: dict[str, Any]) -> list[list[Any]]:
    e = _edges(ds)
    return _top([(a, len(targets)) for a, targets in e.out.items()])


_SNB_REFERENCES: dict[str, Callable[[SnbGraph, dict[str, Any]], list[list[Any]]]] = {
    "pattern_triangle": _triangle,
    "pattern_square": _square,
    "pattern_common": _common,
    "pattern_join": _join,
    "pattern_triangle_hub": _triangle,
    "pattern_triangle_all": _triangle_all,
}

_GRAPH_REFERENCES: dict[str, Callable[[Dataset, dict[str, Any]], list[list[Any]]]] = {
    "pm_two_hop_count": _two_hop_count,
    "pm_two_hop_top10": _two_hop_top10,
    "agg_out_degree_top10": _out_degree_top10,
    "sem_trail_self_loop": _two_hop_count,
}

_ORDERED = {qid: ordered for qid, (_, ordered) in _BASE.items()}


def reference(query_id: str, graph: Any, binding: dict[str, Any]) -> list[list[Any]]:
    """The reference answer of one query for one binding. `graph` is the `SnbGraph` of an SNB item and the `Dataset`
    itself of a Graphalytics or multigraph item."""
    if query_id in _SNB_REFERENCES:
        rows = _SNB_REFERENCES[query_id](graph, binding)
    elif query_id in _GRAPH_REFERENCES:
        rows = _GRAPH_REFERENCES[query_id](graph, binding)
    else:
        raise KeyError(f"unknown pattern query {query_id!r}")
    return canonical(rows, ordered=_ORDERED[query_id])


# Bindings --------------------------------------------------------------------------------------------------


def _degrees(g: SnbGraph) -> dict[str, int]:
    return {p: len(friends) for p, friends in _neighbours(g).items() if friends}


def _at_most_median(g: SnbGraph) -> list[str]:
    """The persons with friends whose KNOWS degree is at most the median, id-sorted."""
    degree = _degrees(g)
    median = sorted(degree.values())[len(degree) // 2]
    return sorted(p for p, d in degree.items() if d <= median)


def _hubs(g: SnbGraph) -> list[str]:
    """The top 1% of all persons by KNOWS degree (ties by id, at least one), id-sorted."""
    degree = _degrees(g)
    size = max(len(g.nodes["Person"]) // HUB_SHARE, 1)
    return sorted(sorted(degree, key=lambda p: (-degree[p], p))[:size])


def _pairs(g: SnbGraph, n: int) -> list[dict[str, Any]]:
    """Pairs of persons with friends half the id-sorted list apart, as IC13's, with the first person spread over the
    first half of the list: IC13 spreads it over the whole list, which repeats each pair, swapped, in the second half
    of its bindings. Never one person twice."""
    persons = _with_friends(g)
    half = len(persons) // 2
    if half == 0:
        raise ValueError("common friends need two persons with friends")
    return [{"person1Id": persons[i], "person2Id": persons[i + half]} for i in _spread(range(half), n)]


def _cuts(ds: Dataset, n: int) -> list[dict[str, Any]]:
    """`n` cuts: cut i is one above the last vertex id of the first (i + 1) n-ths of the id-sorted vertices."""
    vertices = _edges(ds).vertices
    n = min(n, len(vertices))
    return [{"cut": vertices[(i + 1) * len(vertices) // n - 1] + 1} for i in range(n)]


def bindings(query_id: str, graph: Any, n: int) -> list[dict[str, Any]]:
    """`n` deterministic bindings (see the module docstring), or one empty binding."""
    if query_id in ("pattern_triangle", "pattern_square"):
        return [{"personId": p} for p in _spread(_at_most_median(graph), n)]
    if query_id == "pattern_triangle_hub":
        return [{"personId": p} for p in _spread(_hubs(graph), n)]
    if query_id == "pattern_common":
        return _pairs(graph, n)
    if query_id == "pm_two_hop_top10":
        return _cuts(graph, n)
    if query_id in _SNB_REFERENCES or query_id in _GRAPH_REFERENCES:
        return [{}]
    raise KeyError(f"unknown pattern query {query_id!r}")
