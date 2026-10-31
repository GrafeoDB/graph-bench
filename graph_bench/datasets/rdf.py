"""The RDF mapping of a dataset, shared by every RDF loader so that all RDF stores hold the same triples.

Mapping (research-b, "Oxigraph"):

- a node is the IRI `<BASE><Label>/<id>` (the label keeps ids unique across labels) with the triple
  `<node> rdf:type v:<Label>` and one triple `<node> v:<column> <literal>` per non-empty column, the `id` column
  included (as `v:id`, so queries can match and return ids as literals);
- an edge is the triple `<src> v:<TYPE> <dst>`, its endpoints built the same way from the `src` and `dst`
  columns and the edge table's source and target labels;
- literals are typed by the column type: INT64 as `xsd:long`, DOUBLE as `xsd:double`, STRING as a simple literal
  (`xsd:string`). An empty CSV field gives no triple (the dataset contract: an empty field means no property).

Edge properties (`KNOWS.creationDate`, `HAS_MEMBER.joinDate`, `STUDY_AT.classYear`, `WORK_AT.workFrom`) are not
mapped yet: a plain triple cannot carry them, and the queries that run on RDF (SNB IS1, IS4, IS6) do not read
them. Mapping them (RDF 1.2 triple terms or an edge node) is decided when a query needs them.

The mapping yields neutral `Triple`s; `ntriples` writes one as an N-Triples line (Grafeo's `INSERT DATA` chunks),
and an adapter with its own term types (pyoxigraph) converts `Iri` and `Literal` itself. `bind` inlines a query's
parameters as literals, `python_value` converts a result literal back to the Python type the references use, and
`COUNT_NODES` / `COUNT_EDGES` are the SPARQL load check.
"""

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from graph_bench.datasets.schema import ColumnType, Dataset

BASE = "http://graph-bench.example/"
VOCAB = BASE + "v/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
XSD = "http://www.w3.org/2001/XMLSchema#"
XSD_LONG = XSD + "long"
XSD_DOUBLE = XSD + "double"
XSD_STRING = XSD + "string"

# The SPARQL prologue of every query over the mapping.
PREFIXES = f"PREFIX v: <{VOCAB}>\nPREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"

_DATATYPES: dict[ColumnType, str] = {"INT64": XSD_LONG, "DOUBLE": XSD_DOUBLE, "STRING": XSD_STRING}

_INTEGERS = {XSD + t for t in ("long", "integer", "int", "short", "byte", "nonNegativeInteger", "positiveInteger")}
_FLOATS = {XSD + t for t in ("double", "float", "decimal")}


@dataclass(frozen=True, slots=True)
class Iri:
    value: str


@dataclass(frozen=True, slots=True)
class Literal:
    """A literal: `lexical` is its lexical form, `datatype` its datatype IRI (`XSD_STRING` for a simple literal)."""

    lexical: str
    datatype: str


Triple = tuple[str, str, Iri | Literal]  # subject IRI, predicate IRI, object


def node_iri(label: str, node_id: object) -> str:
    """The IRI of the node `node_id` with label `label`."""
    return f"{BASE}{label}/{quote(str(node_id), safe='')}"


def term(name: str) -> str:
    """The vocabulary IRI of a label, column or edge type: `v:<name>`."""
    return VOCAB + name


def _lexical(value: object, column_type: ColumnType) -> str:
    if column_type == "DOUBLE":
        number = float(value)  # type: ignore[arg-type]
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "INF" if number > 0 else "-INF"
        return repr(number)
    return str(value)


def literal(value: object, column_type: ColumnType) -> Literal:
    """The literal of a column value of type `column_type`."""
    return Literal(_lexical(value, column_type), _DATATYPES[column_type])


def triples(dataset: Dataset) -> Iterator[Triple]:
    """Every triple of the dataset, nodes first (table by table), then edges, in CSV order."""
    rdf_type = RDF_TYPE
    for table in dataset.nodes:
        type_iri = Iri(term(table.label))
        types = {c.name: c.type for c in table.columns}
        predicates = {c.name: term(c.name) for c in table.columns}
        for row in dataset.rows(table):
            subject = node_iri(table.label, row["id"])
            yield subject, rdf_type, type_iri
            for name, value in row.items():
                if value != "":
                    yield subject, predicates[name], literal(value, types[name])
    for edge_table in dataset.edges:
        predicate = term(edge_table.type)
        for row in dataset.rows(edge_table):
            src = node_iri(edge_table.src, row["src"])
            yield src, predicate, Iri(node_iri(edge_table.dst, row["dst"]))


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


def _term_text(obj: Iri | Literal) -> str:
    if isinstance(obj, Iri):
        return f"<{obj.value}>"
    if obj.datatype == XSD_STRING:
        return f'"{_escape(obj.lexical)}"'
    return f'"{_escape(obj.lexical)}"^^<{obj.datatype}>'


def ntriples(triple: Triple) -> str:
    """One triple as an N-Triples line (without the newline); also valid inside a SPARQL `INSERT DATA` block."""
    subject, predicate, obj = triple
    return f"<{subject}> <{predicate}> {_term_text(obj)} ."


def _param_literal(param: object) -> Literal:
    if isinstance(param, bool):
        return Literal("true" if param else "false", XSD + "boolean")
    if isinstance(param, int):
        return literal(param, "INT64")
    if isinstance(param, float):
        return literal(param, "DOUBLE")
    if isinstance(param, str):
        return literal(param, "STRING")
    raise TypeError(f"no RDF literal for a parameter of type {type(param).__name__}")


_PARAM = re.compile(r"\$(\w+)")


def bind(text: str, params: Mapping[str, Any]) -> str:
    """`text` with each `$name` replaced by the literal of `params[name]`, typed as the mapping types values.

    SPARQL has no standard query parameters; the RDF queries put each parameter in an inline `VALUES` block,
    `VALUES ?personId { $personId }`, and every RDF adapter binds them with this function, so all RDF stores run
    the same text. Strings become simple literals (escaped), ints `xsd:long` and floats `xsd:double`, so they join
    with the loaded literals. A `$name` without a parameter raises `KeyError`.
    """
    return _PARAM.sub(lambda m: _term_text(_param_literal(params[m.group(1)])), text)


# The load check of an RDF store (`Database.counts`): every node has exactly one `rdf:type` triple, and every
# edge is one triple with an IRI object other than `rdf:type` (the dataset has no duplicate edges, which a set of
# triples would merge).
COUNT_NODES = PREFIXES + "SELECT (COUNT(?s) AS ?c) WHERE { ?s rdf:type ?t }"
COUNT_EDGES = PREFIXES + "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o . FILTER(isIRI(?o) && ?p != rdf:type) }"


def python_value(lexical: str, datatype: object) -> int | float | str:
    """A result literal as the Python value the references use: integers to int, doubles to float, else str.

    `datatype` is the datatype IRI, or an IRI object whose `str()` is it (Grafeo's `grafeo.terms.IRI` does not
    hash like its string), or None.
    """
    iri = None if datatype is None else str(datatype)
    if iri in _INTEGERS:
        return int(lexical)
    if iri in _FLOATS:
        return float(lexical)
    return str(lexical)
