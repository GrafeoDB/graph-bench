"""Hops: 1, 2 and 3 hop expansions and a variable-length expansion over KNOWS in either direction.

Four openCypher queries (ported from the legacy `traversal.py` hop_1, hop_2 and BFS depth 3), each counting the
distinct persons a pattern reaches from one person, with deterministic bindings and a Python reference answer, and
their twins in SQL/PGQ (`<id>_sql`, all four) and SPARQL (`<id>_rdf`, all but `hop_var`), which ask the same
question through the other models and share the openCypher item's bindings and references (`same_as`). The twins
are rows of this category, so DuckDB + DuckPGQ and Oxigraph appear in Hops too. IC13 (shortest path, `snb.py`) is
in Hops as well and stays there.

- `hop_1`: the friends of the person.
- `hop_2`: the persons at the end of a 2-edge walk from the person, the person excluded.
- `hop_3`: the persons at the end of a 3-edge walk `s-a-b-f` from the person, with `b <> s`, `f <> a` and `f <> s`.
- `hop_var`: the persons within 3 hops (`*1..3`), the person excluded.

Match semantics. openCypher matches a pattern under relationship isomorphism (no edge twice in one match); some
engines match walks (homomorphism, an edge may repeat). The texts make both give the same answer:

- The generated data has no repeated KNOWS pair and no self-loop (`_adjacency` checks both and raises otherwise),
  so an edge is identified by its two persons.
- In `s-a-f` the two edges are the same exactly when `f = s`, which `hop_2` excludes.
- In `s-a-b-f` the first two edges are the same exactly when `b = s`, the last two exactly when `f = a`, and the
  first and the last exactly when `b = s` (or `b = a`, a self-loop); `hop_3` excludes `b = s` and `f = a`, so
  every walk it keeps is a trail. `f <> s` is the condition of the question itself: a trail may close a triangle
  back at the person.
- `*1..3` reaches the radius-3 ball around the person under either semantics: a shortest path never repeats an
  edge, and every walk end within 3 edges is within distance 3.

The references enumerate the walks with explicit edge identities (relationship isomorphism) and apply the same
conditions, so they are also the homomorphic answer. `count(DISTINCT f.id)` gives one integer row per binding,
compared exactly (`rel_eps` 0).

Bindings: `selection.bindings` persons with friends, spread evenly over the id-sorted list (`snb._spread`), which
starts at `person_0`, one of the generator's influencers (hubs, about twenty times the median degree on
snb-small).

The SQL/PGQ texts follow ruling R6 as `snb_sql.py` does: `GRAPH_TABLE (snb MATCH ...)`, the graph pattern WHERE
after the pattern with the openCypher conditions, `$personId`, and a variable on every edge pattern
(`-[k:KNOWS]-`). `hop_var` is the bounded quantifier `-[k:KNOWS]-{1,3}`, which the standard allows without a path
selector (only an unbounded quantifier needs one). The distinct count is the outer SQL query's, over the `COLUMNS`
rows. The SPARQL texts follow `snb_rdf.py` (an inline `VALUES` block first, bound by `rdf.bind`): a basic graph
pattern with a named variable for every person of the openCypher pattern, each step the property path
`(v:KNOWS|^v:KNOWS)` (KNOWS in either direction), and the same conditions as FILTERs over the ids. SPARQL matches
walks, which the conditions make equal to the openCypher answer. There is no SPARQL `hop_var`: SPARQL 1.1 has no
bounded repetition of a property path (`{1,3}` was dropped from the 1.1 drafts,
https://www.w3.org/TR/sparql11-query/#propertypaths), and a union of the three fixed-length patterns would be a
different query text, not the shared one; Oxigraph shows ⊘ on the `hop_var` row by model.

HelixDB (dialect "helix") has no query text: its text is the name of a builder in
`graph_bench.databases.helix_queries`, or a reason the DSL cannot express the query.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from graph_bench.datasets.rdf import PREFIXES
from graph_bench.datasets.snb import SnbGraph
from graph_bench.suites import Query
from graph_bench.suites.snb import _spread, _with_friends

# The openCypher items, in category order.
HOP_IDS = ("hop_1", "hop_2", "hop_3", "hop_var")

# One distinct count per binding: compared exactly.
_EXACT = 0.0

_HOP_1 = """\
MATCH (s:Person {id: $personId})-[:KNOWS]-(f:Person)
RETURN count(DISTINCT f.id) AS persons"""

_HOP_2 = """\
MATCH (s:Person {id: $personId})-[:KNOWS]-(:Person)-[:KNOWS]-(f:Person)
WHERE f.id <> $personId
RETURN count(DISTINCT f.id) AS persons"""

_HOP_3 = """\
MATCH (s:Person {id: $personId})-[:KNOWS]-(a:Person)-[:KNOWS]-(b:Person)-[:KNOWS]-(f:Person)
WHERE b.id <> $personId AND f.id <> a.id AND f.id <> $personId
RETURN count(DISTINCT f.id) AS persons"""

_HOP_VAR = """\
MATCH (s:Person {id: $personId})-[:KNOWS*1..3]-(f:Person)
WHERE f.id <> $personId
RETURN count(DISTINCT f.id) AS persons"""

_HOP_1_SQL = """\
SELECT count(DISTINCT fid) AS persons
FROM GRAPH_TABLE (snb
  MATCH (s:Person)-[k:KNOWS]-(f:Person)
  WHERE s.id = $personId
  COLUMNS (f.id AS fid)
)"""

_HOP_2_SQL = """\
SELECT count(DISTINCT fid) AS persons
FROM GRAPH_TABLE (snb
  MATCH (s:Person)-[k1:KNOWS]-(m:Person)-[k2:KNOWS]-(f:Person)
  WHERE s.id = $personId AND f.id <> $personId
  COLUMNS (f.id AS fid)
)"""

_HOP_3_SQL = """\
SELECT count(DISTINCT fid) AS persons
FROM GRAPH_TABLE (snb
  MATCH (s:Person)-[k1:KNOWS]-(a:Person)-[k2:KNOWS]-(b:Person)-[k3:KNOWS]-(f:Person)
  WHERE s.id = $personId AND b.id <> $personId AND f.id <> a.id AND f.id <> $personId
  COLUMNS (f.id AS fid)
)"""

_HOP_VAR_SQL = """\
SELECT count(DISTINCT fid) AS persons
FROM GRAPH_TABLE (snb
  MATCH (s:Person)-[k:KNOWS]-{1,3}(f:Person)
  WHERE s.id = $personId AND f.id <> $personId
  COLUMNS (f.id AS fid)
)"""

_HOP_1_RDF = f"""\
{PREFIXES}SELECT (COUNT(DISTINCT ?fid) AS ?persons) WHERE {{
  VALUES ?personId {{ $personId }}
  ?s a v:Person ; v:id ?personId ; (v:KNOWS|^v:KNOWS) ?f .
  ?f a v:Person ; v:id ?fid .
}}"""

_HOP_2_RDF = f"""\
{PREFIXES}SELECT (COUNT(DISTINCT ?fid) AS ?persons) WHERE {{
  VALUES ?personId {{ $personId }}
  ?s a v:Person ; v:id ?personId ; (v:KNOWS|^v:KNOWS) ?m .
  ?m a v:Person ; (v:KNOWS|^v:KNOWS) ?f .
  ?f a v:Person ; v:id ?fid .
  FILTER(?fid != ?personId)
}}"""

_HOP_3_RDF = f"""\
{PREFIXES}SELECT (COUNT(DISTINCT ?fid) AS ?persons) WHERE {{
  VALUES ?personId {{ $personId }}
  ?s a v:Person ; v:id ?personId ; (v:KNOWS|^v:KNOWS) ?a .
  ?a a v:Person ; v:id ?aid ; (v:KNOWS|^v:KNOWS) ?b .
  ?b a v:Person ; v:id ?bid ; (v:KNOWS|^v:KNOWS) ?f .
  ?f a v:Person ; v:id ?fid .
  FILTER(?bid != ?personId && ?fid != ?aid && ?fid != ?personId)
}}"""

# (id, title, openCypher, SQL/PGQ, SPARQL or None)
_ITEMS: tuple[tuple[str, str, str, str, str | None], ...] = (
    ("hop_1", "1-hop", _HOP_1, _HOP_1_SQL, _HOP_1_RDF),
    ("hop_2", "2-hop", _HOP_2, _HOP_2_SQL, _HOP_2_RDF),
    ("hop_3", "3-hop", _HOP_3, _HOP_3_SQL, _HOP_3_RDF),
    ("hop_var", "Variable length 1 to 3", _HOP_VAR, _HOP_VAR_SQL, None),
)

# The openCypher items, then their SQL/PGQ twins, then their SPARQL twins: the category's rows in that order.
QUERIES: tuple[Query, ...] = (
    *(
        Query(qid, title, "hops", "property_graph", "snb", {"cypher": cypher, "helix": qid}, False, _EXACT)
        for qid, title, cypher, _, _ in _ITEMS
    ),
    *(
        Query(f"{qid}_sql", f"{title} (SQL/PGQ)", "hops", "sql_pgq", "snb", {"sql_pgq": sql}, False, _EXACT, qid)
        for qid, title, _, sql, _ in _ITEMS
    ),
    *(
        Query(f"{qid}_rdf", f"{title} (SPARQL)", "hops", "rdf", "snb", {"sparql": sparql}, False, _EXACT, qid)
        for qid, title, _, _, sparql in _ITEMS
        if sparql is not None
    ),
)

_BY_ID = {q.id: q for q in QUERIES}


# References ------------------------------------------------------------------------------------------------

Adjacency = dict[str, list[tuple[str, int]]]

# The adjacency of the last graph asked for (held, so its identity cannot be reused): the references of one graph
# are computed binding after binding.
_CACHED: tuple[SnbGraph, Adjacency] | None = None


def _adjacency(g: SnbGraph) -> Adjacency:
    """Person -> (other person, edge number) over every KNOWS edge, in both directions.

    Raises ValueError on a self-loop or a repeated pair (in either direction): the texts' conditions equal edge
    distinctness only on a simple graph, so a generator change must fail here rather than change the semantics.
    """
    global _CACHED
    if _CACHED is not None and _CACHED[0] is g:
        return _CACHED[1]
    adj: Adjacency = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    edge = 0
    for src, edges in g.out.get("KNOWS", {}).items():
        for dst, _ in edges:
            if src == dst:
                raise ValueError(f"KNOWS self-loop on {src}: the hop conditions assume none")
            pair = (min(src, dst), max(src, dst))
            if pair in seen:
                raise ValueError(f"repeated KNOWS pair {pair}: the hop conditions assume unique pairs")
            seen.add(pair)
            adj[src].append((dst, edge))
            adj[dst].append((src, edge))
            edge += 1
    _CACHED = (g, adj)
    return adj


def _hop_1(adj: Adjacency, s: str) -> set[str]:
    return {f for f, _ in adj.get(s, [])}


def _hop_2(adj: Adjacency, s: str) -> set[str]:
    found: set[str] = set()
    for a, e1 in adj.get(s, []):
        for f, e2 in adj[a]:
            if e2 != e1 and f != s:
                found.add(f)
    return found


def _hop_3(adj: Adjacency, s: str) -> set[str]:
    found: set[str] = set()
    for a, e1 in adj.get(s, []):
        for b, e2 in adj[a]:
            if e2 == e1 or b == s:
                continue
            for f, e3 in adj[b]:
                if e3 not in (e1, e2) and f != a and f != s:
                    found.add(f)
    return found


def _hop_var(adj: Adjacency, s: str) -> set[str]:
    """The ends of the 1, 2 and 3 edge trails from `s` (no edge twice), without `s`."""
    found: set[str] = set()
    for a, e1 in adj.get(s, []):
        found.add(a)
        for b, e2 in adj[a]:
            if e2 == e1:
                continue
            found.add(b)
            for f, e3 in adj[b]:
                if e3 not in (e1, e2):
                    found.add(f)
    found.discard(s)
    return found


_REFERENCES: dict[str, Callable[[Adjacency, str], set[str]]] = {
    "hop_1": _hop_1,
    "hop_2": _hop_2,
    "hop_3": _hop_3,
    "hop_var": _hop_var,
}


def reference(query_id: str, graph: SnbGraph, binding: dict[str, Any]) -> list[list[Any]]:
    """The reference answer of one query for one binding: one row, the number of distinct persons reached."""
    if query_id not in _REFERENCES:
        raise KeyError(f"unknown hop query {query_id!r}")
    return [[len(_REFERENCES[query_id](_adjacency(graph), binding["personId"]))]]


# Bindings --------------------------------------------------------------------------------------------------


def bindings(query_id: str, graph: SnbGraph, n: int) -> list[dict[str, Any]]:
    """`n` persons with friends, spread evenly over the id-sorted list (hubs included)."""
    if query_id not in _REFERENCES:
        raise KeyError(f"unknown hop query {query_id!r}")
    return [{"personId": p} for p in _spread(_with_friends(graph), n)]
