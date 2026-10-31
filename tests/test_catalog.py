import re
from pathlib import Path

from typer.testing import CliRunner

from graph_bench.cli import app
from graph_bench.suites import catalog, snb_rdf, snb_sql
from graph_bench.suites.catalog import Case, cases
from graph_bench.suites.snb import QUERIES
from graph_bench.types import STANDARD_DIALECT


def _item(item_id: str) -> catalog.Query | catalog.Algorithm:
    return next(i for i in catalog.ITEMS if i.id == item_id)


def test_push_selection_cases() -> None:
    result = cases("push", ["grafeo", "ladybug"])
    assert len(result) == 2 * (13 + 4 + 3)
    assert all(c.threads == 1 for c in result)
    assert all(c.scale == "small" for c in result)
    assert [c.bench_id for c in result] == sorted(c.bench_id for c in result)


def test_full_selection_single_threaded() -> None:
    result = cases("full", ["grafeo", "ladybug"])
    assert result
    assert all(c.threads == 1 for c in result)


def test_bench_id_format() -> None:
    is1 = _item("snb_is1")
    case = Case(is1, "ladybug", "small", 1)
    assert case.bench_id == "benches/test_complex_queries.py::test_snb_is1[ladybug-small]"
    assert case.test_name == "test_snb_is1"
    assert Case(_item("snb_ic13"), "ladybug", "small", 1).bench_id.startswith("benches/test_hops.py::")
    assert Case(is1, "ladybug", "small", None).bench_id.endswith("[ladybug-small-allcores]")


def test_list_output_is_stable() -> None:
    runner = CliRunner()
    first = runner.invoke(app, ["list", "--selection", "push", "--db", "grafeo,ladybug"])
    second = runner.invoke(app, ["list", "--selection", "push", "--db", "grafeo,ladybug"])
    assert first.exit_code == 0, first.output
    lines = first.output.splitlines()
    assert lines == sorted(lines)
    assert first.output == second.output
    expected = sorted(
        [c.bench_id for c in cases("push", ["grafeo", "ladybug"])] + catalog.load_ids("push", ["grafeo", "ladybug"])
    )
    assert lines == expected
    assert "benches/test_load.py::test_load_snb[grafeo-small]" in lines
    assert "benches/test_load.py::test_load_graphalytics[ladybug-small]" in lines
    # Grafeo's RDF store is a load of its own; LadybugDB serves no RDF, so it has none.
    assert "benches/test_load.py::test_load_snb_rdf[grafeo-small]" in lines
    assert "benches/test_load.py::test_load_snb_rdf[ladybug-small]" not in lines
    assert "benches/test_sql_pgq.py::test_snb_sql_is1[ladybug-small]" in lines
    assert "benches/test_rdf.py::test_snb_rdf_is6[grafeo-small]" in lines


def test_limited_queries_have_total_order() -> None:
    pattern = re.compile(r"(\w+\.)?\w*[iI]d\s+(ASC|DESC)?\s*LIMIT")
    limited = [q for q in QUERIES if "LIMIT" in q.text["cypher"]]
    assert limited
    for q in limited:
        for dialect, text in q.text.items():
            if dialect == "helix":  # the name of a DSL builder, not query text
                assert text == q.id
                continue
            assert "ORDER BY" in text, q.id
            assert pattern.search(text), q.id


def test_every_query_has_cypher_text() -> None:
    assert len(QUERIES) == 10
    for q in QUERIES:
        assert q.text["cypher"].strip(), q.id
        assert q.model == "property_graph"
        assert q.dataset == "snb"


def test_push_and_full_include_sql_pgq_and_rdf() -> None:
    for name in ("push", "full"):
        assert {"sql_pgq", "rdf"} <= catalog.selection(name).categories


def test_model_queries_share_the_snb_references() -> None:
    """SQL/PGQ and RDF ask the openCypher queries' questions: same bindings, reference file and row order."""
    by_id = {item.id: item for item in catalog.ITEMS}
    queries = [*snb_sql.QUERIES, *snb_rdf.QUERIES]
    assert [q.id for q in queries] == [
        "snb_sql_is1",
        "snb_sql_is4",
        "snb_sql_is6",
        "snb_sql_ic13",
        "snb_rdf_is1",
        "snb_rdf_is4",
        "snb_rdf_is6",
    ]
    for q in queries:
        shared = by_id[q.same_as]
        assert q.reference_id == shared.id == q.id.replace("_sql", "").replace("_rdf", "")
        assert (q.dataset, q.ordered, q.rel_eps, q.title) == (
            shared.dataset,
            shared.ordered,
            shared.rel_eps,
            shared.title,
        )
        assert q.category == q.model
        assert q.text[STANDARD_DIALECT[q.model]].strip()
        assert catalog.reference_path(q, "small", base=Path("b")) == catalog.reference_path(
            shared, "small", base=Path("b")
        )
    # One reference file per shared item, written once.
    assert [i.id for i in catalog.reference_items("push")] == [
        i.id for i in catalog.ITEMS if not getattr(i, "same_as", "")
    ]


def test_sql_pgq_texts_name_the_graph_and_use_element_where() -> None:
    for q in snb_sql.QUERIES:
        text = q.text["sql_pgq"]
        assert "GRAPH_TABLE (snb\n" in text, q.id
        assert "WHERE" in text.split("MATCH", 1)[1].split("COLUMNS", 1)[0], q.id
        assert "{id:" not in text, q.id


def test_sparql_texts_take_parameters_as_values() -> None:
    for q in snb_rdf.QUERIES:
        text = q.text["sparql"]
        group = text.split("WHERE {", 1)[1].lstrip()
        assert group.startswith("VALUES ?"), q.id  # the parameter block comes first in the group


def test_text_for_falls_back_to_the_models_standard_text() -> None:
    sql = snb_sql.QUERIES[0]
    assert sql.text_for("duckpgq") == sql.text["sql_pgq"]
    assert snb_rdf.QUERIES[0].text_for("oxigraph") == snb_rdf.QUERIES[0].text["sparql"]
    assert QUERIES[0].text_for("nope") == QUERIES[0].text["cypher"]


def test_load_ids_per_store() -> None:
    assert (
        catalog.load_id("snb", "grafeo", "small", 1, "rdf") == "benches/test_load.py::test_load_snb_rdf[grafeo-small]"
    )
    assert catalog.load_benchmark("snb") == "load_snb"
    assert catalog.load_benchmark("snb", "rdf") == "load_snb_rdf"
    assert catalog.load_keys("push", frozenset({"rdf"})) == [("snb", "rdf")]
    assert catalog.load_keys("push", frozenset({"sql_pgq"})) == [("snb", "graph")]
    assert catalog.load_keys("push", frozenset({"property_graph", "sql_pgq", "rdf"})) == [
        ("graphalytics", "graph"),
        ("snb", "graph"),
        ("snb", "rdf"),
    ]
