"""LDBC SNB IS1, IS4 and IS6 in SPARQL 1.1 over the RDF mapping (`graph_bench.datasets.rdf`), the RDF category.

They ask the same questions as the openCypher texts in `snb.py` on the same data, so they share those queries'
bindings and reference answers (`same_as`) and return the same columns in the same order.

SPARQL has no standard query parameters, so a parameter is an inline `VALUES` block, first in the group:
`VALUES ?personId { $personId }`, where an RDF adapter replaces `$personId` with the binding's literal through
`rdf.bind` (the same text runs on every RDF store). Results are read by variable name.

- IS4's "Message" is `?m a ?type` over `VALUES ?type { v:Post v:Comment }`; the image file stands in for empty
  content with `OPTIONAL` and `IF(!BOUND(...) ...)`, as the openCypher `CASE` does.
- IS6's `REPLY_OF*0..` is the zero-or-more property path `v:REPLY_OF*`.
- IC13 (shortest path length) is not in this category: SPARQL 1.1 property paths test reachability but give no
  path length.
"""

from graph_bench.answers import SNB_EPS
from graph_bench.datasets.rdf import PREFIXES
from graph_bench.suites import Query

_IS1 = f"""\
{PREFIXES}SELECT ?firstName ?lastName ?birthday ?locationIP ?browserUsed ?cityId ?gender ?creationDate WHERE {{
  VALUES ?personId {{ $personId }}
  ?n a v:Person ; v:id ?personId ;
     v:firstName ?firstName ; v:lastName ?lastName ; v:birthday ?birthday ; v:locationIP ?locationIP ;
     v:browserUsed ?browserUsed ; v:gender ?gender ; v:creationDate ?creationDate ; v:LIVES_IN ?c .
  ?c v:id ?cityId .
}}"""

_IS4 = f"""\
{PREFIXES}SELECT ?messageCreationDate ?messageContent WHERE {{
  VALUES ?messageId {{ $messageId }}
  ?m v:id ?messageId ; v:creationDate ?messageCreationDate .
  VALUES ?type {{ v:Post v:Comment }}
  ?m a ?type .
  OPTIONAL {{ ?m v:content ?content }}
  OPTIONAL {{ ?m v:imageFile ?imageFile }}
  BIND(IF(!BOUND(?content) || ?content = "", ?imageFile, ?content) AS ?messageContent)
}}"""

_IS6 = f"""\
{PREFIXES}SELECT ?forumId ?forumTitle ?moderatorId ?moderatorFirstName ?moderatorLastName WHERE {{
  VALUES ?messageId {{ $messageId }}
  ?m v:id ?messageId ; v:REPLY_OF* ?p .
  ?p a v:Post .
  ?f v:CONTAINER_OF ?p ; v:id ?forumId ; v:title ?forumTitle ; v:HAS_MODERATOR ?mod .
  ?mod v:id ?moderatorId ; v:firstName ?moderatorFirstName ; v:lastName ?moderatorLastName .
}}"""


def _q(qid: str, title: str, text: str, *, ordered: bool) -> Query:
    return Query(f"snb_rdf_{qid}", title, "rdf", "rdf", "snb", {"sparql": text}, ordered, SNB_EPS, f"snb_{qid}")


QUERIES: tuple[Query, ...] = (
    _q("is1", "IS1 person profile", _IS1, ordered=False),
    _q("is4", "IS4 message content", _IS4, ordered=False),
    _q("is6", "IS6 forum of a message", _IS6, ordered=False),
)
