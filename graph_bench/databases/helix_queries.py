"""The SNB queries as HelixDB DSL builders (dialect `helix`).

HelixDB's Python SDK has no query text (https://docs.helix-db.com/database/helix-db/query-guides/reading-data): a
read is an operation tree built with `g()` and sent as a `read_batch`. The `helix` text of a suite query is the
name of a builder here (`snb_is1`, ...). A builder takes the binding and returns the request plus a `rows` function
that lists the answer in the query's column order. `rows` only reads the named columns out of the result
dictionaries (and, for IC13, counts the nodes of the path HelixDB returned); every filter, join, order, limit and
projection runs inside HelixDB.

A query the DSL cannot express maps to a reason string instead of a builder; the adapter raises `NotSupported`
with it (the case shows as not available, see docs/known-disagreements.md). Nothing is computed in Python to make
up for a missing feature.

DSL references: parameters https://docs.helix-db.com/database/helix-db/query-guides/parameters, reading
https://docs.helix-db.com/database/helix-db/query-guides/reading-data, traversals
https://docs.helix-db.com/database/helix-db/query-guides/traversals, filtering
https://docs.helix-db.com/database/helix-db/query-guides/filtering, projections
https://docs.helix-db.com/database/helix-db/query-guides/projections, branching and repeat
https://docs.helix-db.com/database/helix-db/query-guides/advanced. The docs list the step names only; the
signatures and the planner's rules (below, "after bind") are those of helix-db 0.3.4 (`helixdb/dsl.py`) and the
embedded runtime 0.3.3.
"""

from collections.abc import Callable, Mapping
from typing import Any

from helixdb import (
    BindingProjection,
    NodeRef,
    Order,
    Predicate,
    QueryRequest,
    RepeatConfig,
    ShortestPathDirection,
    SourcePredicate,
    define_params,
    g,
    param,
    read_batch,
    sub,
)

Rows = Callable[[Mapping[str, Any]], list[tuple[Any, ...]]]
Builder = Callable[[Mapping[str, Any]], tuple[QueryRequest, Rows]]

# Reply chains are followed to the root post; the shared text has no bound (`*0..`), the DSL needs one
# ("Recursive and fan-out operations must have application-level bounds",
# https://docs.helix-db.com/database/helix-db/query-guides/advanced). The generated data's longest chain has 7
# replies; 50 is far above any chain and is a bound, not a filter.
REPLY_DEPTH = 50
# IC13 `[:KNOWS*]` is unbounded; `shortest_path` needs a maximum depth. Far above the small-world diameter of the
# generated friend graphs, so a bound that never cuts a path.
PATH_DEPTH = 100

_PERSON = define_params({"personId": param.string()})
_MESSAGE = define_params({"messageId": param.string()})
_IC2 = define_params({"personId": param.string(), "maxDate": param.i64()})
_IC13 = define_params({"person1Id": param.string(), "person2Id": param.string()})

_B = BindingProjection


def _content(binding: str | None = None, alias: str = "messageContent") -> BindingProjection:
    """`CASE WHEN content IS NULL OR content = '' THEN imageFile ELSE content END`: empty strings are stored as
    missing properties (see `HelixDatabase.load`), so this is the first of content and imageFile that is present."""
    if binding is None:
        return _B.coalesce([_B.current_ref("content"), _B.current_ref("imageFile")], alias)
    return _B.coalesce([_B.binding_ref(binding, "content"), _B.binding_ref(binding, "imageFile")], alias)


def _columns(entries: tuple[str, ...], columns: tuple[str, ...]) -> Rows:
    """Rows of the named entries, each as the tuple of `columns` (a missing property is None)."""

    def rows(result: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        return [tuple(row.get(c) for c in columns) for e in entries for row in result[e]]

    return rows


def _request(params: Any, values: Mapping[str, Any], **entries: Any) -> QueryRequest:
    batch = read_batch()
    for name, traversal in entries.items():
        batch = batch.var_as(name, traversal)
    return batch.returning(list(entries)).to_query_request(params, dict(values))


def _message(label: str) -> Any:
    """A message by id: the per-label source, because the DSL has no label disjunction in a source predicate and
    the unique `id` index is per label. The Post and Comment entries together are `m:Post OR m:Comment`."""
    return g().n_with_label_where(label, SourcePredicate.eq("id", _MESSAGE.messageId))


def snb_is1(b: Mapping[str, Any]) -> tuple[QueryRequest, Rows]:
    person = g().n_with_label_where("Person", SourcePredicate.eq("id", _PERSON.personId)).bind("n").out("LIVES_IN")
    keys = ("firstName", "lastName", "birthday", "locationIP", "browserUsed")
    projections = [_B.binding("n", k, k) for k in keys] + [_B.current("id", "cityId")]
    projections += [_B.binding("n", k, k) for k in ("gender", "creationDate")]
    columns = (*keys, "cityId", "gender", "creationDate")
    return _request(_PERSON, b, r=person.project_bindings(projections)), _columns(("r",), columns)


def snb_is2(b: Mapping[str, Any]) -> tuple[QueryRequest, Rows]:
    # The top 10 messages are ordered and limited first: the planner rejects `order_by` and `repeat` after `bind`
    # ("order_by is not supported after bind"). The reply chain is then followed with `optional` steps, which keep
    # a node that has no REPLY_OF edge (a Post is its own root, `*0..`); `repeat` cannot run after `bind`.
    t = (
        g()
        .n_with_label_where("Person", SourcePredicate.eq("id", _PERSON.personId))
        .in_("HAS_CREATOR")
        .order_by_multiple([("creationDate", Order.DESC), ("id", Order.DESC)])
        .limit(10)
        .bind("m")
    )
    for _ in range(REPLY_DEPTH):
        t = t.optional(sub().out("REPLY_OF"))
    t = t.bind("post").out("HAS_CREATOR")
    projections = [
        _B.binding("m", "id", "messageId"),
        _content("m"),
        _B.binding("m", "creationDate", "messageCreationDate"),
        _B.binding("post", "id", "originalPostId"),
        _B.current("id", "originalPosterId"),
        _B.current("firstName", "originalPosterFirstName"),
        _B.current("lastName", "originalPosterLastName"),
    ]
    columns = (
        "messageId",
        "messageContent",
        "messageCreationDate",
        "originalPostId",
        "originalPosterId",
        "originalPosterFirstName",
        "originalPosterLastName",
    )
    return _request(_PERSON, b, r=t.project_bindings(projections)), _columns(("r",), columns)


def snb_is4(b: Mapping[str, Any]) -> tuple[QueryRequest, Rows]:
    projections = [_B.current("creationDate", "messageCreationDate"), _content()]
    entries = {label: _message(label).project_bindings(projections) for label in ("Post", "Comment")}
    return _request(_MESSAGE, b, **entries), _columns(("Post", "Comment"), ("messageCreationDate", "messageContent"))


def snb_is5(b: Mapping[str, Any]) -> tuple[QueryRequest, Rows]:
    entries = {
        label: _message(label).out("HAS_CREATOR").value_map(["id", "firstName", "lastName"])
        for label in ("Post", "Comment")
    }
    return _request(_MESSAGE, b, **entries), _columns(("Post", "Comment"), ("id", "firstName", "lastName"))


def snb_is6(b: Mapping[str, Any]) -> tuple[QueryRequest, Rows]:
    # `[:REPLY_OF*0..]` to a Post: a Post is its own root, a Comment follows REPLY_OF until the label is Post.
    root_comment = _message("Comment").repeat(
        RepeatConfig.new(sub().out("REPLY_OF")).until(Predicate.eq("$label", "Post")).max_depth(REPLY_DEPTH)
    )
    projections = [
        _B.binding("f", "id", "forumId"),
        _B.binding("f", "title", "forumTitle"),
        _B.current("id", "moderatorId"),
        _B.current("firstName", "moderatorFirstName"),
        _B.current("lastName", "moderatorLastName"),
    ]
    entries = {
        label: root.in_("CONTAINER_OF").bind("f").out("HAS_MODERATOR").project_bindings(projections)
        for label, root in (("Post", _message("Post")), ("Comment", root_comment))
    }
    columns = ("forumId", "forumTitle", "moderatorId", "moderatorFirstName", "moderatorLastName")
    return _request(_MESSAGE, b, **entries), _columns(("Post", "Comment"), columns)


def snb_ic2(b: Mapping[str, Any]) -> tuple[QueryRequest, Rows]:
    # Friends in either direction, their messages up to maxDate, ordered and limited; the friend is reached again
    # from each message (`order_by` is not supported after `bind`).
    t = (
        g()
        .n_with_label_where("Person", SourcePredicate.eq("id", _IC2.personId))
        .both("KNOWS")
        .in_("HAS_CREATOR")
        .where(Predicate.lte("creationDate", _IC2.maxDate))
        .order_by_multiple([("creationDate", Order.DESC), ("id", Order.ASC)])
        .limit(20)
        .bind("m")
        .out("HAS_CREATOR")
    )
    projections = [
        _B.current("id", "friendId"),
        _B.current("firstName", "friendFirstName"),
        _B.current("lastName", "friendLastName"),
        _B.binding("m", "id", "messageId"),
        _content("m"),
        _B.binding("m", "creationDate", "messageCreationDate"),
    ]
    columns = (
        "friendId",
        "friendFirstName",
        "friendLastName",
        "messageId",
        "messageContent",
        "messageCreationDate",
    )
    return _request(_IC2, b, r=t.project_bindings(projections)), _columns(("r",), columns)


def snb_ic13(b: Mapping[str, Any]) -> tuple[QueryRequest, Rows]:
    # `shortest_path` over KNOWS in both directions returns the nodes of a shortest path (empty when none).
    person1 = g().n_with_label_where("Person", SourcePredicate.eq("id", _IC13.person1Id))
    person2 = g().n_with_label_where("Person", SourcePredicate.eq("id", _IC13.person2Id))
    path = g().shortest_path(
        NodeRef.var("a"), NodeRef.var("b"), PATH_DEPTH, label="KNOWS", direction=ShortestPathDirection.BOTH
    )

    def rows(result: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        nodes = result["p"]
        # length(path) is the number of edges: one less than the nodes. No path is -1 and a node to itself is 0
        # (the single-node path), as in the shared text.
        return [(len(nodes) - 1 if nodes else -1,)]

    return _request(_IC13, b, a=person1, b=person2, p=path), rows


def _unsupported_order(query: str, detail: str) -> str:
    return (
        f"{query}: the shared text's ORDER BY {detail}. The DSL's order_by sorts a stream by the properties of its "
        "own elements, so that key cannot be applied, and the query is not run without it"
    )


BUILDERS: dict[str, Builder | str] = {
    "snb_is1": snb_is1,
    "snb_is2": snb_is2,
    "snb_is3": _unsupported_order(
        "IS3", "friendshipCreationDate DESC, friendId ASC needs the id of the node at the far end of the sorted edge"
    ),
    "snb_is4": snb_is4,
    "snb_is5": snb_is5,
    "snb_is6": snb_is6,
    "snb_is7": _unsupported_order(
        "IS7", "commentCreationDate DESC, replyAuthorId ASC needs the id of the author node of each sorted comment"
    ),
    "snb_ic2": snb_ic2,
    "snb_ic6": (
        "IC6: ranking the tag groups (ORDER BY postCount DESC, otherTagName ASC LIMIT 10) cannot be expressed: "
        "group_count is a terminal step whose result is a map that order_by and limit cannot rank by count"
    ),
    "snb_ic13": snb_ic13,
}
