"""LDBC SNB Interactive v1 reads (IS1-IS7, IC2, IC6, IC13), simplified to the generated schema.

Each query has an openCypher text, deterministic parameter bindings and a Python reference answer computed
from the generated data. "Message" means Post or Comment. A Post with an image has an empty `content`; its
"content" column is the `imageFile` instead (empty strings and NULL are treated the same, since CSV loaders
differ on which one an empty field becomes).
"""

from collections import Counter, deque
from collections.abc import Callable, Sequence
from typing import Any

from graph_bench.answers import SNB_EPS, canonical
from graph_bench.datasets.snb import SnbGraph
from graph_bench.suites import Query
from graph_bench.types import Category

_CONTENT = "CASE WHEN m.content IS NULL OR m.content = '' THEN m.imageFile ELSE m.content END"

_IS1 = """\
MATCH (n:Person {id: $personId})-[:LIVES_IN]->(c:City)
RETURN n.firstName AS firstName, n.lastName AS lastName, n.birthday AS birthday,
       n.locationIP AS locationIP, n.browserUsed AS browserUsed, c.id AS cityId,
       n.gender AS gender, n.creationDate AS creationDate"""

_IS2 = f"""\
MATCH (:Person {{id: $personId}})<-[:HAS_CREATOR]-(m)
WITH m
ORDER BY m.creationDate DESC, m.id DESC
LIMIT 10
MATCH (m)-[:REPLY_OF*0..]->(post:Post)-[:HAS_CREATOR]->(op:Person)
RETURN m.id AS messageId, {_CONTENT} AS messageContent, m.creationDate AS messageCreationDate,
       post.id AS originalPostId, op.id AS originalPosterId,
       op.firstName AS originalPosterFirstName, op.lastName AS originalPosterLastName
ORDER BY messageCreationDate DESC, messageId DESC"""

_IS3 = """\
MATCH (n:Person {id: $personId})-[k:KNOWS]-(friend:Person)
RETURN friend.id AS friendId, friend.firstName AS firstName, friend.lastName AS lastName,
       k.creationDate AS friendshipCreationDate
ORDER BY friendshipCreationDate DESC, friendId ASC"""

_IS4 = f"""\
MATCH (m {{id: $messageId}})
WHERE m:Post OR m:Comment
RETURN m.creationDate AS messageCreationDate, {_CONTENT} AS messageContent"""

# NeuG (dialect "neug") reuses the two LadybugDB texts below, and only these: its parser rejects `WHERE m:Post OR
# m:Comment` (IS4) and `shortestPath(...)` (IC13). A multi-label pattern `(m:Post:Comment)` means "either label" in
# NeuG as well (https://neug.io/docs/cypher_manual/query_clauses/match_clause/), and `[:KNOWS* SHORTEST]` is its
# shortest-path form, so both keep the meaning of the shared text.
_IS4_LADYBUG = f"""\
MATCH (m:Post:Comment {{id: $messageId}})
RETURN m.creationDate AS messageCreationDate, {_CONTENT} AS messageContent"""

_IS5 = """\
MATCH (m {id: $messageId})-[:HAS_CREATOR]->(p:Person)
RETURN p.id AS creatorId, p.firstName AS firstName, p.lastName AS lastName"""

_IS6_RETURN = """\
RETURN f.id AS forumId, f.title AS forumTitle, mod.id AS moderatorId,
       mod.firstName AS moderatorFirstName, mod.lastName AS moderatorLastName"""

_IS6 = f"""\
MATCH (m {{id: $messageId}})-[:REPLY_OF*0..]->(p:Post)<-[:CONTAINER_OF]-(f:Forum)-[:HAS_MODERATOR]->(mod:Person)
{_IS6_RETURN}"""

_IS6_LADYBUG = f"""\
MATCH (m:Post:Comment {{id: $messageId}})-[:REPLY_OF*0..]->(p:Post)<-[:CONTAINER_OF]-(f:Forum)
MATCH (f)-[:HAS_MODERATOR]->(mod:Person)
{_IS6_RETURN}"""

_IS7 = """\
MATCH (m:Post {id: $messageId})-[:HAS_CREATOR]->(author:Person)
MATCH (m)<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person)
OPTIONAL MATCH (author)-[k:KNOWS]-(p)
RETURN c.id AS commentId, c.content AS commentContent, c.creationDate AS commentCreationDate,
       p.id AS replyAuthorId, p.firstName AS replyAuthorFirstName, p.lastName AS replyAuthorLastName,
       k IS NOT NULL AS replyAuthorKnowsOriginalAuthor
ORDER BY commentCreationDate DESC, replyAuthorId ASC"""

_IC2 = f"""\
MATCH (:Person {{id: $personId}})-[:KNOWS]-(friend:Person)<-[:HAS_CREATOR]-(m)
WHERE m.creationDate <= $maxDate
RETURN friend.id AS friendId, friend.firstName AS friendFirstName, friend.lastName AS friendLastName,
       m.id AS messageId, {_CONTENT} AS messageContent, m.creationDate AS messageCreationDate
ORDER BY messageCreationDate DESC, messageId ASC
LIMIT 20"""

_IC6 = """\
MATCH (:Person {id: $personId})-[:KNOWS*1..2]-(friend:Person)
WHERE friend.id <> $personId
WITH DISTINCT friend
MATCH (friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(:Tag {name: $tagName})
MATCH (post)-[:HAS_TAG]->(t:Tag)
WHERE t.name <> $tagName
WITH t.name AS otherTagName, t.id AS tagId, count(DISTINCT post) AS postCount
ORDER BY postCount DESC, otherTagName ASC, tagId ASC
LIMIT 10
RETURN otherTagName, postCount
ORDER BY postCount DESC, otherTagName ASC"""

_IC13_RETURN = (
    "RETURN CASE WHEN p1.id = p2.id THEN 0 WHEN path IS NULL THEN -1 ELSE length(path) END AS shortestPathLength"
)

_IC13 = f"""\
MATCH (p1:Person {{id: $person1Id}}), (p2:Person {{id: $person2Id}})
OPTIONAL MATCH path = shortestPath((p1)-[:KNOWS*]-(p2))
{_IC13_RETURN}"""

_IC13_LADYBUG = f"""\
MATCH (p1:Person {{id: $person1Id}}), (p2:Person {{id: $person2Id}})
OPTIONAL MATCH path = (p1)-[:KNOWS* SHORTEST]-(p2)
{_IC13_RETURN}"""


def _q(qid: str, title: str, text: dict[str, str], *, ordered: bool, category: Category = "complex_queries") -> Query:
    return Query(qid, title, category, "property_graph", "snb", text, ordered, SNB_EPS)


QUERIES: tuple[Query, ...] = (
    _q("snb_is1", "IS1 person profile", {"cypher": _IS1}, ordered=False),
    _q("snb_is2", "IS2 recent messages of a person", {"cypher": _IS2}, ordered=True),
    _q("snb_is3", "IS3 friends of a person", {"cypher": _IS3}, ordered=True),
    _q(
        "snb_is4", "IS4 message content", {"cypher": _IS4, "ladybug": _IS4_LADYBUG, "neug": _IS4_LADYBUG}, ordered=False
    ),
    _q("snb_is5", "IS5 message creator", {"cypher": _IS5}, ordered=False),
    _q("snb_is6", "IS6 forum of a message", {"cypher": _IS6, "ladybug": _IS6_LADYBUG}, ordered=False),
    # The ORDER BY of IS7 leaves ties (same author, same day), so rows are compared as a multiset.
    _q("snb_is7", "IS7 replies of a message", {"cypher": _IS7}, ordered=False),
    _q("snb_ic2", "IC2 recent messages by friends", {"cypher": _IC2}, ordered=True),
    _q("snb_ic6", "IC6 tag co-occurrence", {"cypher": _IC6}, ordered=True),
    _q(
        "snb_ic13",
        "IC13 shortest path",
        {"cypher": _IC13, "ladybug": _IC13_LADYBUG, "neug": _IC13_LADYBUG},
        ordered=False,
        category="hops",
    ),
)

_BY_ID = {q.id: q for q in QUERIES}


# Graph helpers ---------------------------------------------------------------------------------------------


def _one(adj: dict[str, list[tuple[str, dict[str, Any]]]], node: str) -> str:
    return adj[node][0][0]


def _friends(g: SnbGraph, pid: str) -> dict[str, int]:
    """Friend id -> friendship creationDate, over KNOWS in either direction."""
    result: dict[str, int] = {}
    for adj in (g.out.get("KNOWS", {}), g.incoming.get("KNOWS", {})):
        for other, props in adj.get(pid, []):
            result[other] = props["creationDate"]
    return result


def _message(g: SnbGraph, mid: str) -> dict[str, Any]:
    return g.nodes["Post"].get(mid) or g.nodes["Comment"][mid]


def _content(props: dict[str, Any]) -> Any:
    return props["content"] if props.get("content") else props.get("imageFile")


def _created(g: SnbGraph, pid: str) -> list[str]:
    return [m for m, _ in g.incoming.get("HAS_CREATOR", {}).get(pid, [])]


def _root_post(g: SnbGraph, mid: str) -> str:
    while mid not in g.nodes["Post"]:
        mid = _one(g.out["REPLY_OF"], mid)
    return mid


def _person(g: SnbGraph, pid: str) -> dict[str, Any]:
    return g.nodes["Person"][pid]


def _messages(g: SnbGraph) -> list[str]:
    return sorted([*g.nodes["Post"], *g.nodes["Comment"]])


# References ------------------------------------------------------------------------------------------------


def _is1(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    p = _person(g, b["personId"])
    city = _one(g.out["LIVES_IN"], b["personId"])
    keys = ("firstName", "lastName", "birthday", "locationIP", "browserUsed")
    return [[*(p[k] for k in keys), city, p["gender"], p["creationDate"]]]


def _is2(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    mids = sorted(_created(g, b["personId"]), key=lambda m: (_message(g, m)["creationDate"], m), reverse=True)
    rows = []
    for mid in mids[:10]:
        msg = _message(g, mid)
        post = _root_post(g, mid)
        op = _one(g.out["HAS_CREATOR"], post)
        opp = _person(g, op)
        rows.append([mid, _content(msg), msg["creationDate"], post, op, opp["firstName"], opp["lastName"]])
    return rows


def _is3(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    friends = sorted(_friends(g, b["personId"]).items(), key=lambda kv: (-kv[1], kv[0]))
    return [[f, _person(g, f)["firstName"], _person(g, f)["lastName"], date] for f, date in friends]


def _is4(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    msg = _message(g, b["messageId"])
    return [[msg["creationDate"], _content(msg)]]


def _is5(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    pid = _one(g.out["HAS_CREATOR"], b["messageId"])
    return [[pid, _person(g, pid)["firstName"], _person(g, pid)["lastName"]]]


def _is6(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    forum = _one(g.incoming["CONTAINER_OF"], _root_post(g, b["messageId"]))
    mod = _one(g.out["HAS_MODERATOR"], forum)
    title = g.nodes["Forum"][forum]["title"]
    return [[forum, title, mod, _person(g, mod)["firstName"], _person(g, mod)["lastName"]]]


def _is7(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    post = b["messageId"]
    author = _one(g.out["HAS_CREATOR"], post)
    author_friends = _friends(g, author)
    rows = []
    for cid, _ in g.incoming.get("REPLY_OF", {}).get(post, []):
        c = g.nodes["Comment"][cid]
        p = _one(g.out["HAS_CREATOR"], cid)
        pp = _person(g, p)
        rows.append([cid, c["content"], c["creationDate"], p, pp["firstName"], pp["lastName"], p in author_friends])
    return rows


def _ic2(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for f in _friends(g, b["personId"]):
        fp = _person(g, f)
        for mid in _created(g, f):
            msg = _message(g, mid)
            if msg["creationDate"] <= b["maxDate"]:
                rows.append([f, fp["firstName"], fp["lastName"], mid, _content(msg), msg["creationDate"]])
    rows.sort(key=lambda r: (-r[5], r[3]))
    return rows[:20]


def _circle(g: SnbGraph, pid: str) -> set[str]:
    """Friends and friends of friends, without the person."""
    friends = set(_friends(g, pid))
    circle = set(friends)
    for f in friends:
        circle.update(_friends(g, f))
    circle.discard(pid)
    return circle


def _circle_posts(g: SnbGraph, pid: str) -> list[list[str]]:
    """Tag names of each post created by the person's circle."""
    tag_names = {tid: t["name"] for tid, t in g.nodes["Tag"].items()}
    has_tag = g.out.get("HAS_TAG", {})
    return [
        [tag_names[t] for t, _ in has_tag.get(mid, [])]
        for member in _circle(g, pid)
        for mid in _created(g, member)
        if mid in g.nodes["Post"]
    ]


def _ic6(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    counts: Counter[str] = Counter()
    for tags in _circle_posts(g, b["personId"]):
        if b["tagName"] in tags:
            counts.update(t for t in tags if t != b["tagName"])
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [[name, n] for name, n in ranked[:10]]


def _ic13(g: SnbGraph, b: dict[str, Any]) -> list[list[Any]]:
    src, dst = b["person1Id"], b["person2Id"]
    if src == dst:
        return [[0]]
    depth = {src: 0}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for other in _friends(g, node):
            if other not in depth:
                if other == dst:
                    return [[depth[node] + 1]]
                depth[other] = depth[node] + 1
                queue.append(other)
    return [[-1]]


_REFERENCES: dict[str, Callable[[SnbGraph, dict[str, Any]], list[list[Any]]]] = {
    "snb_is1": _is1,
    "snb_is2": _is2,
    "snb_is3": _is3,
    "snb_is4": _is4,
    "snb_is5": _is5,
    "snb_is6": _is6,
    "snb_is7": _is7,
    "snb_ic2": _ic2,
    "snb_ic6": _ic6,
    "snb_ic13": _ic13,
}


def reference(query_id: str, graph: SnbGraph, binding: dict[str, Any]) -> list[list[Any]]:
    """The canonical reference answer of one query for one binding."""
    return canonical(_REFERENCES[query_id](graph, binding), ordered=_BY_ID[query_id].ordered)


# Bindings --------------------------------------------------------------------------------------------------


def _spread[T](candidates: Sequence[T], n: int) -> list[T]:
    """`n` evenly spaced elements of `candidates` (all of them if there are fewer)."""
    if len(candidates) <= n:
        return list(candidates)
    return [candidates[i * len(candidates) // n] for i in range(n)]


def _with_friends(g: SnbGraph) -> list[str]:
    return sorted(p for p in g.nodes["Person"] if _friends(g, p))


def _ic2_bindings(g: SnbGraph, n: int) -> list[dict[str, Any]]:
    candidates = [p for p in _with_friends(g) if any(_created(g, f) for f in _friends(g, p))]
    result = []
    for pid in _spread(candidates, n):
        dates = sorted(_message(g, m)["creationDate"] for f in _friends(g, pid) for m in _created(g, f))
        result.append({"personId": pid, "maxDate": dates[len(dates) // 2]})
    return result


def _ic6_bindings(g: SnbGraph, n: int) -> list[dict[str, Any]]:
    result = []
    for pid in _spread(_with_friends(g), n):
        counts = Counter(t for tags in _circle_posts(g, pid) for t in tags)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        tag = ranked[0][0] if ranked else min(t["name"] for t in g.nodes["Tag"].values())
        result.append({"personId": pid, "tagName": tag})
    return result


def _ic13_bindings(g: SnbGraph, n: int) -> list[dict[str, Any]]:
    """Pairs of persons with friends half the id-sorted list apart, plus LDBC IC13's two edge cases.

    The LDBC definition returns -1 when no path exists and 0 when person1 = person2, so the last bindings are a
    person and a person without friends (when the data has one) and a person with itself.
    """
    persons = _with_friends(g)
    lonely = sorted(p for p in g.nodes["Person"] if not _friends(g, p))
    extra = [{"person1Id": persons[0], "person2Id": lonely[0]}] if lonely else []
    extra.append({"person1Id": persons[len(persons) // 2], "person2Id": persons[len(persons) // 2]})
    half = len(persons) // 2
    starts = _spread(range(len(persons)), max(n - len(extra), 0))
    pairs = [{"person1Id": persons[i], "person2Id": persons[(i + half) % len(persons)]} for i in starts]
    return pairs + extra


def bindings(query_id: str, graph: SnbGraph, n: int) -> list[dict[str, Any]]:
    """`n` deterministic parameter bindings, evenly spaced over the id-sorted candidates."""
    g = graph
    if query_id == "snb_is1":
        return [{"personId": p} for p in _spread(sorted(g.out["LIVES_IN"]), n)]
    if query_id == "snb_is2":
        return [{"personId": p} for p in _spread(sorted(g.incoming["HAS_CREATOR"]), n)]
    if query_id == "snb_is3":
        return [{"personId": p} for p in _spread(_with_friends(g), n)]
    if query_id in ("snb_is4", "snb_is5", "snb_is6"):
        return [{"messageId": m} for m in _spread(_messages(g), n)]
    if query_id == "snb_is7":
        replied = sorted(p for p in g.incoming["REPLY_OF"] if p in g.nodes["Post"])
        return [{"messageId": m} for m in _spread(replied, n)]
    if query_id == "snb_ic2":
        return _ic2_bindings(g, n)
    if query_id == "snb_ic6":
        return _ic6_bindings(g, n)
    if query_id == "snb_ic13":
        return _ic13_bindings(g, n)
    raise KeyError(f"unknown SNB query {query_id!r}")
