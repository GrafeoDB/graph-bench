"""Scans: whole-label reads of the SNB data, with a filter, an aggregation, or an order plus limit.

Five openCypher queries (ported from the legacy `query.py` and `storage.py` scans), each with deterministic
bindings and a Python reference answer, and their twins in SQL/PGQ (`<id>_sql`) and SPARQL (`<id>_rdf`), which ask
the same question through the other models and share the openCypher item's bindings and references (`same_as`).
The twins are rows of this category, so DuckDB + DuckPGQ and Oxigraph appear in Scans too.

- `scan_label`: the id of every Comment; unordered.
- `scan_filter`: the Comments created in a window `[from, to)`; unordered. Each window covers about 10% of the
  comments by creationDate, and the windows are spread evenly over the comments.
- `scan_aggregate`: Persons grouped by browser and gender, with their count, earliest birthday and latest
  creationDate; ordered by the two keys (every group is one row, so the order is total).
- `scan_order_limit`: the 20 newest Comments of at least a length, with their content (strings with commas and
  apostrophes, which makes it the Load (query language) content probe, `catalog.LOAD_QL_PROBES`); ordered by
  creationDate descending, then id, so the order is total. The bindings are the distinct comment lengths.
- `scan_filter_avg`: the number and the average length of the Comments longer than a threshold; one row. The
  thresholds are spread over the integers from one below the shortest comment to one below the longest, so every
  binding counts at least one comment (an average over no rows is NULL in openCypher and SQL but 0 in SPARQL).

The SQL/PGQ texts follow ruling R6 as `snb_sql.py` does: `GRAPH_TABLE (snb MATCH ...)`, the graph pattern WHERE
after the pattern, `$name` parameters (none of these patterns has an edge, so the rule that every edge pattern has a
variable does not arise). The aggregation, order and limit are the outer SQL query's, over the `COLUMNS` rows. The
SPARQL texts follow `snb_rdf.py`: each parameter is an inline `VALUES` block, first in the group, bound by
`rdf.bind`. Their triple patterns require each property they read: the generated data gives every Person all four
properties read here and every Comment a content, a creationDate and a length, so the answers are the openCypher
ones.

HelixDB (dialect "helix") has no query text: its text is the name of a builder in
`graph_bench.databases.helix_queries`, or a reason the DSL cannot express the query.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from graph_bench.answers import SNB_EPS, canonical
from graph_bench.datasets.rdf import PREFIXES
from graph_bench.datasets.snb import SnbGraph
from graph_bench.suites import Query
from graph_bench.suites.snb import _spread

# Rows of the order plus limit query.
LIMIT = 20

_LABEL = "MATCH (m:Comment) RETURN m.id AS id"

_FILTER = """\
MATCH (m:Comment)
WHERE m.creationDate >= $from AND m.creationDate < $to
RETURN m.id AS id"""

_AGGREGATE = """\
MATCH (p:Person)
RETURN p.browserUsed AS browser, p.gender AS gender, count(*) AS persons, min(p.birthday) AS firstBirthday,
       max(p.creationDate) AS lastCreated
ORDER BY browser, gender"""

_ORDER_LIMIT = f"""\
MATCH (m:Comment)
WHERE m.length >= $minLength
RETURN m.id AS id, m.content AS content, m.creationDate AS created
ORDER BY created DESC, id ASC
LIMIT {LIMIT}"""

_FILTER_AVG = """\
MATCH (c:Comment)
WHERE c.length > $min
RETURN count(*) AS n, avg(c.length) AS mean"""

_LABEL_SQL = """\
SELECT id
FROM GRAPH_TABLE (snb
  MATCH (m:Comment)
  COLUMNS (m.id AS id)
)"""

_FILTER_SQL = """\
SELECT id
FROM GRAPH_TABLE (snb
  MATCH (m:Comment)
  WHERE m.creationDate >= $from AND m.creationDate < $to
  COLUMNS (m.id AS id)
)"""

_AGGREGATE_SQL = """\
SELECT browser, gender, count(*) AS persons, min(birthday) AS firstBirthday, max(creationDate) AS lastCreated
FROM GRAPH_TABLE (snb
  MATCH (p:Person)
  COLUMNS (p.browserUsed AS browser, p.gender AS gender, p.birthday AS birthday, p.creationDate AS creationDate)
)
GROUP BY browser, gender
ORDER BY browser, gender"""

_ORDER_LIMIT_SQL = f"""\
SELECT id, content, created
FROM GRAPH_TABLE (snb
  MATCH (m:Comment)
  WHERE m.length >= $minLength
  COLUMNS (m.id AS id, m.content AS content, m.creationDate AS created)
)
ORDER BY created DESC, id ASC
LIMIT {LIMIT}"""

_FILTER_AVG_SQL = """\
SELECT count(*) AS n, avg(len) AS mean
FROM GRAPH_TABLE (snb
  MATCH (c:Comment)
  WHERE c.length > $min
  COLUMNS (c.length AS len)
)"""

_LABEL_RDF = f"""\
{PREFIXES}SELECT ?id WHERE {{
  ?m a v:Comment ; v:id ?id .
}}"""

_FILTER_RDF = f"""\
{PREFIXES}SELECT ?id WHERE {{
  VALUES ?from {{ $from }}
  VALUES ?to {{ $to }}
  ?m a v:Comment ; v:id ?id ; v:creationDate ?created .
  FILTER(?created >= ?from && ?created < ?to)
}}"""

_AGGREGATE_RDF = f"""\
{PREFIXES}SELECT ?browser ?gender (COUNT(*) AS ?persons) (MIN(?birthday) AS ?firstBirthday)
       (MAX(?creationDate) AS ?lastCreated) WHERE {{
  ?p a v:Person ; v:browserUsed ?browser ; v:gender ?gender ; v:birthday ?birthday ; v:creationDate ?creationDate .
}}
GROUP BY ?browser ?gender
ORDER BY ?browser ?gender"""

_ORDER_LIMIT_RDF = f"""\
{PREFIXES}SELECT ?id ?content ?created WHERE {{
  VALUES ?minLength {{ $minLength }}
  ?m a v:Comment ; v:id ?id ; v:content ?content ; v:creationDate ?created ; v:length ?length .
  FILTER(?length >= ?minLength)
}}
ORDER BY DESC(?created) ASC(?id)
LIMIT {LIMIT}"""

_FILTER_AVG_RDF = f"""\
{PREFIXES}SELECT (COUNT(*) AS ?n) (AVG(?length) AS ?mean) WHERE {{
  VALUES ?min {{ $min }}
  ?c a v:Comment ; v:length ?length .
  FILTER(?length > ?min)
}}"""

# (id, title, openCypher, SQL/PGQ, SPARQL, ordered)
_ITEMS: tuple[tuple[str, str, str, str, str, bool], ...] = (
    ("scan_label", "Label scan", _LABEL, _LABEL_SQL, _LABEL_RDF, False),
    ("scan_filter", "Property filter", _FILTER, _FILTER_SQL, _FILTER_RDF, False),
    ("scan_aggregate", "Aggregation", _AGGREGATE, _AGGREGATE_SQL, _AGGREGATE_RDF, True),
    ("scan_order_limit", "Order plus limit", _ORDER_LIMIT, _ORDER_LIMIT_SQL, _ORDER_LIMIT_RDF, True),
    ("scan_filter_avg", "Property filter with average", _FILTER_AVG, _FILTER_AVG_SQL, _FILTER_AVG_RDF, False),
)

# The openCypher items, then their SQL/PGQ twins, then their SPARQL twins: the category's rows in that order.
QUERIES: tuple[Query, ...] = (
    *(
        Query(qid, title, "scans", "property_graph", "snb", {"cypher": cypher, "helix": qid}, ordered, SNB_EPS)
        for qid, title, cypher, _, _, ordered in _ITEMS
    ),
    *(
        Query(f"{qid}_sql", f"{title} (SQL/PGQ)", "scans", "sql_pgq", "snb", {"sql_pgq": sql}, ordered, SNB_EPS, qid)
        for qid, title, _, sql, _, ordered in _ITEMS
    ),
    *(
        Query(f"{qid}_rdf", f"{title} (SPARQL)", "scans", "rdf", "snb", {"sparql": sparql}, ordered, SNB_EPS, qid)
        for qid, title, _, _, sparql, ordered in _ITEMS
    ),
)

_BY_ID = {q.id: q for q in QUERIES}


# References ------------------------------------------------------------------------------------------------


def _comments(g: SnbGraph) -> list[dict[str, Any]]:
    return list(g.nodes["Comment"].values())


def _label(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    return [[c["id"]] for c in _comments(g)]


def _filter(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    return [[c["id"]] for c in _comments(g) if b["from"] <= c["creationDate"] < b["to"]]


def _aggregate(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in g.nodes["Person"].values():
        groups[(p["browserUsed"], p["gender"])].append(p)
    return [
        [browser, gender, len(ps), min(p["birthday"] for p in ps), max(p["creationDate"] for p in ps)]
        for (browser, gender), ps in sorted(groups.items())
    ]


def _order_limit(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    long_enough = [c for c in _comments(g) if c["length"] >= b["minLength"]]
    long_enough.sort(key=lambda c: (-c["creationDate"], c["id"]))
    return [[c["id"], c["content"], c["creationDate"]] for c in long_enough[:LIMIT]]


def _filter_avg(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    lengths = [c["length"] for c in _comments(g) if c["length"] > b["min"]]
    return [[len(lengths), sum(lengths) / len(lengths) if lengths else None]]


_REFERENCES: dict[str, Callable[[SnbGraph, dict[str, Any]], list[list[Any]]]] = {
    "scan_label": _label,
    "scan_filter": _filter,
    "scan_aggregate": _aggregate,
    "scan_order_limit": _order_limit,
    "scan_filter_avg": _filter_avg,
}


def reference(query_id: str, graph: SnbGraph, binding: dict[str, Any]) -> list[list[Any]]:
    """The canonical reference answer of one query for one binding."""
    return canonical(_REFERENCES[query_id](graph, binding), ordered=_BY_ID[query_id].ordered)


# Bindings --------------------------------------------------------------------------------------------------


def _windows(g: SnbGraph, n: int) -> list[dict[str, Any]]:
    """`n` windows `[from, to)` of `len // 10` consecutive comments in creationDate order, evenly spaced.

    A window starts at a comment's date and ends (exclusive) at the date of the comment `len // 10` places later, so
    it holds about 10% of the comments (ties at either end move a few in or out).
    """
    dates = sorted(c["creationDate"] for c in _comments(g))
    width = max(len(dates) // 10, 1)
    starts = _spread(range(len(dates) - width), n)
    return [{"from": dates[s], "to": dates[s + width]} for s in starts]


def _lengths(g: SnbGraph) -> list[int]:
    return sorted({c["length"] for c in _comments(g)})


def bindings(query_id: str, graph: SnbGraph, n: int) -> list[dict[str, Any]]:
    """`n` deterministic parameter bindings (fewer where the data has fewer distinct values)."""
    if query_id in ("scan_label", "scan_aggregate"):
        return [{}]
    if query_id == "scan_filter":
        return _windows(graph, n)
    if query_id == "scan_order_limit":
        return [{"minLength": x} for x in _spread(_lengths(graph), n)]
    if query_id == "scan_filter_avg":
        lengths = _lengths(graph)
        return [{"min": t} for t in _spread(range(lengths[0] - 1, lengths[-1]), n)]
    raise KeyError(f"unknown scan query {query_id!r}")
