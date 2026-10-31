"""The shared RDF mapping (`graph_bench.datasets.rdf`): triples, N-Triples lines, parameter binding, literals."""

from pathlib import Path

import pytest

from graph_bench.datasets import cache, rdf
from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable

XSD_LONG = "http://www.w3.org/2001/XMLSchema#long"


def _tiny(tmp_path: Path) -> Dataset:
    (tmp_path / "Person.csv").write_text(
        'id,name,age,score\np1,"Al ""the"" \\ one",30,1.5\np 2,,41,\n', encoding="utf-8", newline="\n"
    )
    (tmp_path / "KNOWS.csv").write_text("src,dst,since\np1,p 2,2020\n", encoding="utf-8", newline="\n")
    nodes = (
        NodeTable(
            "Person",
            (Column("id", "STRING"), Column("name", "STRING"), Column("age", "INT64"), Column("score", "STRING")),
            "Person.csv",
        ),
    )
    edges = (EdgeTable("KNOWS", "Person", "Person", (Column("since", "INT64"),), "KNOWS.csv"),)
    return Dataset("tiny", "small", tmp_path, nodes, edges, 2, 1)


def test_triples_follow_the_mapping(tmp_path: Path) -> None:
    triples = list(rdf.triples(_tiny(tmp_path)))
    p1, p2 = "http://graph-bench.example/Person/p1", "http://graph-bench.example/Person/p%202"
    v = "http://graph-bench.example/v/"
    assert (p1, rdf.RDF_TYPE, rdf.Iri(v + "Person")) in triples
    assert (p1, v + "id", rdf.Literal("p1", rdf.XSD_STRING)) in triples
    assert (p1, v + "age", rdf.Literal("30", XSD_LONG)) in triples
    # An empty field gives no triple; the edge's property is not mapped.
    assert not [t for t in triples if t[0] == p2 and t[1] in (v + "name", v + "score")]
    assert (p1, v + "KNOWS", rdf.Iri(p2)) in triples
    assert not [t for t in triples if t[1] == v + "since"]
    assert len(triples) == 2 + 2 + 3 + 1 + 1  # 2 types, 2 ids, name/age/score of p1, age of p2, 1 edge


def test_ntriples_escapes_literals(tmp_path: Path) -> None:
    lines = [rdf.ntriples(t) for t in rdf.triples(_tiny(tmp_path))]
    assert (
        '<http://graph-bench.example/Person/p1> <http://graph-bench.example/v/name> "Al \\"the\\" \\\\ one" .' in lines
    )
    assert f'<http://graph-bench.example/Person/p1> <http://graph-bench.example/v/age> "30"^^<{XSD_LONG}> .' in lines
    assert all(line.endswith(" .") for line in lines)


def test_bind_inlines_typed_literals() -> None:
    text = "VALUES ?a { $a } VALUES ?b { $b } VALUES ?c { $c } VALUES ?d { $d }"
    bound = rdf.bind(text, {"a": 'x "y"', "b": 7, "c": 0.5, "d": True})
    assert bound == (
        'VALUES ?a { "x \\"y\\"" } '
        f'VALUES ?b {{ "7"^^<{XSD_LONG}> }} '
        'VALUES ?c { "0.5"^^<http://www.w3.org/2001/XMLSchema#double> } '
        'VALUES ?d { "true"^^<http://www.w3.org/2001/XMLSchema#boolean> }'
    )
    with pytest.raises(KeyError):
        rdf.bind("VALUES ?x { $missing }", {})
    with pytest.raises(TypeError):
        rdf.bind("VALUES ?x { $x }", {"x": [1]})


class _Iri(str):
    """An IRI type that hashes unlike its text, as `grafeo.terms.IRI` does."""

    __hash__ = object.__hash__


def test_python_value_converts_by_datatype() -> None:
    assert rdf.python_value("42", XSD_LONG) == 42
    assert isinstance(rdf.python_value("42", "http://www.w3.org/2001/XMLSchema#integer"), int)
    assert rdf.python_value("1.5", "http://www.w3.org/2001/XMLSchema#double") == 1.5
    assert rdf.python_value("x", rdf.XSD_STRING) == "x"
    assert rdf.python_value("x", None) == "x"
    assert rdf.python_value("42", _Iri(XSD_LONG)) == 42


def test_double_literals() -> None:
    assert rdf.literal(1.5, "DOUBLE").lexical == "1.5"
    assert rdf.literal(float("inf"), "DOUBLE").lexical == "INF"
    assert rdf.literal(float("-inf"), "DOUBLE").lexical == "-INF"
    assert rdf.literal(float("nan"), "DOUBLE").lexical == "NaN"


def test_snb_mapping_counts_match_the_dataset(push_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every node has one rdf:type triple and every edge one IRI-valued triple: the RDF load check's premise."""
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    ds = cache.prepare("push")[("snb", "small")]
    triples = list(rdf.triples(ds))
    types = [t for t in triples if t[1] == rdf.RDF_TYPE]
    edges = [t for t in triples if isinstance(t[2], rdf.Iri) and t[1] != rdf.RDF_TYPE]
    assert len({t[0] for t in types}) == len(types) == ds.node_count
    assert len(set(edges)) == len(edges) == ds.edge_count
