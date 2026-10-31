"""LDBC SNB IS1, IS4, IS6 and IC13 in SQL/PGQ (ISO/IEC 9075-16), the SQL/PGQ category.

They ask the same questions as the openCypher texts in `snb.py` on the same data, so they share those queries'
bindings and reference answers (`same_as`) and return the same columns in the same order.

The texts are standard SQL/PGQ, written once for every engine (dialect "sql_pgq"); an engine that cannot run a
form gets a marker, not a weaker text:

- `GRAPH_TABLE (snb MATCH ...)` names the property graph, as the standard requires; a database loads the SNB
  dataset as a property graph named `snb`.
- A node is filtered by an element pattern WHERE clause, `(n:Person WHERE n.id = $personId)`: the SQL/PGQ form of
  the openCypher property map `{id: $personId}` (SQL/PGQ has no property maps in element patterns), and the form
  that filters before a path is selected. Every edge pattern has a variable (`-[l:LIVES_IN]->`).
- "Message" is the label disjunction `Post|Comment`, as the openCypher texts' `m:Post OR m:Comment`.
- IS6's `REPLY_OF*0..` is the quantified edge `-[r:REPLY_OF]->*`. The standard allows an unbounded quantifier only
  in a selective path pattern (or under a restrictive path mode), so the pattern is `ANY SHORTEST`: a reply chain
  is a tree path (every comment replies to exactly one message), so there is exactly one path from a message to
  its forum's moderator and `ANY SHORTEST` selects that same path; the answer is the openCypher one.
- IC13 is `ANY SHORTEST` over KNOWS in either direction (`-[k:KNOWS]-+`) with `PATH_LENGTH`. `MIN` over the
  matches always gives one row (NULL when there is no path, turned into -1), and the same person gives 0, as
  LDBC IC13 and the openCypher text define.
- Parameters are named, `$personId`, bound from the binding's dict.
"""

from graph_bench.answers import SNB_EPS
from graph_bench.suites import Query

_IS1 = """\
SELECT firstName, lastName, birthday, locationIP, browserUsed, cityId, gender, creationDate
FROM GRAPH_TABLE (snb
  MATCH (n:Person WHERE n.id = $personId)-[l:LIVES_IN]->(c:City)
  COLUMNS (n.firstName AS firstName, n.lastName AS lastName, n.birthday AS birthday,
           n.locationIP AS locationIP, n.browserUsed AS browserUsed, c.id AS cityId,
           n.gender AS gender, n.creationDate AS creationDate)
)"""

_IS4 = """\
SELECT messageCreationDate,
       CASE WHEN content IS NULL OR content = '' THEN imageFile ELSE content END AS messageContent
FROM GRAPH_TABLE (snb
  MATCH (m:Post|Comment WHERE m.id = $messageId)
  COLUMNS (m.creationDate AS messageCreationDate, m.content AS content, m.imageFile AS imageFile)
)"""

_IS6 = """\
SELECT forumId, forumTitle, moderatorId, moderatorFirstName, moderatorLastName
FROM GRAPH_TABLE (snb
  MATCH ANY SHORTEST (m:Post|Comment WHERE m.id = $messageId)-[r:REPLY_OF]->*(p:Post)
        <-[co:CONTAINER_OF]-(f:Forum)-[hm:HAS_MODERATOR]->(mod:Person)
  COLUMNS (f.id AS forumId, f.title AS forumTitle, mod.id AS moderatorId,
           mod.firstName AS moderatorFirstName, mod.lastName AS moderatorLastName)
)"""

_IC13 = """\
SELECT CASE WHEN $person1Id = $person2Id THEN 0 ELSE COALESCE(MIN(len), -1) END AS shortestPathLength
FROM GRAPH_TABLE (snb
  MATCH path = ANY SHORTEST (p1:Person WHERE p1.id = $person1Id)-[k:KNOWS]-+(p2:Person WHERE p2.id = $person2Id)
  COLUMNS (PATH_LENGTH(path) AS len)
)"""


def _q(qid: str, title: str, text: str, *, ordered: bool) -> Query:
    return Query(
        f"snb_sql_{qid}", title, "sql_pgq", "sql_pgq", "snb", {"sql_pgq": text}, ordered, SNB_EPS, f"snb_{qid}"
    )


QUERIES: tuple[Query, ...] = (
    _q("is1", "IS1 person profile", _IS1, ordered=False),
    _q("is4", "IS4 message content", _IS4, ordered=False),
    _q("is6", "IS6 forum of a message", _IS6, ordered=False),
    _q("ic13", "IC13 shortest path", _IC13, ordered=False),
)
