import json
import re
from pathlib import Path

from typer.testing import CliRunner

from graph_bench.cli import app
from graph_bench.suites import GRAPH_MODELS, Write, catalog, snb_rdf, snb_sql
from graph_bench.suites.catalog import Case, cases
from graph_bench.suites.snb import QUERIES
from graph_bench.types import BUDGET_EXEMPT, CATEGORY_NAMES, CATEGORY_ORDER, LOAD_CATEGORIES, STANDARD_DIALECT


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
    assert lines == catalog.all_ids("push", ["grafeo", "ladybug"])
    assert "benches/test_load.py::test_load_snb[grafeo-small]" in lines
    assert "benches/test_load.py::test_load_graphalytics[ladybug-small]" in lines
    # Grafeo's RDF store is a load of its own. LadybugDB serves no RDF: its RDF load is planned too, and reported ⊘.
    assert "benches/test_load.py::test_load_snb_rdf[grafeo-small]" in lines
    assert "benches/test_load.py::test_load_snb_rdf[ladybug-small]" in lines
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


def test_sql_pgq_texts_name_the_graph_and_filter_in_the_graph_pattern_where() -> None:
    """Ruling R6: the filter is the graph pattern WHERE after the pattern, never inside an element pattern."""
    for q in snb_sql.QUERIES:
        text = q.text["sql_pgq"]
        assert "GRAPH_TABLE (snb\n" in text, q.id
        pattern, rest = text.split("MATCH", 1)[1].split("COLUMNS", 1)[0].split("WHERE", 1)
        assert re.match(r"\s+\w+\.id = \$\w+", rest), q.id  # one graph pattern WHERE, right after the pattern
        assert not re.search(r"\([^()]*\bWHERE\b[^()]*\)", text.split("COLUMNS", 1)[0]), q.id
        assert pattern.rstrip().endswith(")"), q.id  # the WHERE follows the last element pattern
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
    # An SQL/PGQ-only database reads the graph store, so it gets the Graphalytics load for the algorithms too.
    assert catalog.load_keys("push", frozenset({"sql_pgq"})) == [("graphalytics", "graph"), ("snb", "graph")]
    assert catalog.all_load_keys("push") == {
        ("graphalytics", "graph"): "property_graph",
        ("snb", "graph"): "property_graph",
        ("snb", "rdf"): "rdf",
    }
    assert catalog.load_keys("push", frozenset({"property_graph", "sql_pgq", "rdf"})) == [
        ("graphalytics", "graph"),
        ("snb", "graph"),
        ("snb", "rdf"),
    ]


def test_algorithms_run_on_any_graph_model() -> None:
    """Graphalytics items need a model that reads a property graph: property graph or SQL/PGQ, not RDF."""
    algorithms = [i for i in catalog.ITEMS if isinstance(i, catalog.Algorithm)]
    assert algorithms
    for algo in algorithms:
        assert algo.models == GRAPH_MODELS == frozenset({"property_graph", "sql_pgq"})
        assert algo.store == "graph"
        assert algo.supported_by(frozenset({"sql_pgq"}))
        assert algo.supported_by(frozenset({"property_graph"}))
        assert not algo.supported_by(frozenset({"rdf"}))
    for q in [*QUERIES, *snb_sql.QUERIES, *snb_rdf.QUERIES]:
        assert q.models == frozenset({q.model})
        assert q.supported_by(frozenset({q.model}))
        assert not q.supported_by(frozenset({"property_graph", "sql_pgq", "rdf"} - {q.model}))


def test_category_order_starts_with_startup_and_loads() -> None:
    assert CATEGORY_ORDER[:3] == ("startup", "load", "load_ql")
    assert CATEGORY_ORDER[3:8] == ("batch_writes", "scans", "hops", "pattern_matching", "complex_queries")
    assert CATEGORY_NAMES["startup"] == "Startup"
    assert CATEGORY_NAMES["load_ql"] == "Load (query language)"
    assert frozenset({"load", "load_ql"}) == LOAD_CATEGORIES
    assert frozenset({"startup", "load", "load_ql"}) == BUDGET_EXEMPT
    assert set(CATEGORY_NAMES) == set(CATEGORY_ORDER)


def test_startup_id_format() -> None:
    assert catalog.STARTUP_SCALE == "empty"
    assert catalog.startup_id("fake", 1) == "benches/test_startup.py::test_startup[fake-empty]"
    assert catalog.startup_id("fake", None) == "benches/test_startup.py::test_startup[fake-empty-allcores]"
    assert catalog.startup_ids("push", ["ladybug", "grafeo"]) == [
        "benches/test_startup.py::test_startup[grafeo-empty]",
        "benches/test_startup.py::test_startup[ladybug-empty]",
    ]


def test_load_ql_ids_cover_the_graph_store_datasets() -> None:
    assert catalog.load_ql_benchmark("snb") == "load_ql_snb"
    assert catalog.load_ql_id("snb", "grafeo", "small", 1) == "benches/test_load_ql.py::test_load_ql_snb[grafeo-small]"
    # Graph-store datasets only: the RDF store has no query-language load row. Every database gets them, as loads.
    assert catalog.load_ql_ids("push", ["oxigraph"]) == [
        "benches/test_load_ql.py::test_load_ql_graphalytics[oxigraph-small]",
        "benches/test_load_ql.py::test_load_ql_snb[oxigraph-small]",
    ]
    assert catalog.load_ql_keys("push") == {"graphalytics": "property_graph", "snb": "property_graph"}
    assert catalog.LOAD_QL_PROBES == {}


def test_selections_plan_startup_load_ql_and_the_core_categories() -> None:
    push, full = catalog.selection("push"), catalog.selection("full")
    assert {"startup", "load_ql", "batch_writes", "scans", "pattern_matching"} <= push.categories
    assert (push.startup_iterations, full.startup_iterations) == (3, 5)
    assert push.resources and full.resources
    assert full.categories == frozenset(CATEGORY_ORDER)


def test_all_ids_is_what_list_prints() -> None:
    ids = catalog.all_ids("push", ["grafeo"])
    assert ids == sorted(
        [c.bench_id for c in cases("push", ["grafeo"])]
        + catalog.load_ids("push", ["grafeo"])
        + catalog.startup_ids("push", ["grafeo"])
        + catalog.load_ql_ids("push", ["grafeo"])
    )
    assert "benches/test_startup.py::test_startup[grafeo-empty]" in ids
    assert "benches/test_load_ql.py::test_load_ql_snb[grafeo-small]" in ids


def test_each_item_has_a_reference_source() -> None:
    from graph_bench.suites import graphalytics, snb

    assert catalog.reference_source("snb_is1") is snb
    assert catalog.reference_source("snb_sql_is1") is snb  # a twin's references come from the item it shares
    assert catalog.reference_source("graphalytics_bfs") is graphalytics


_WRITE = Write(
    "w",
    "W",
    "batch_writes",
    "property_graph",
    "snb",
    run={"cypher": "RUN", "ladybug": "RUN L"},
    check={"cypher": "CHECK"},
    setup={"cypher": ("S1", "S2")},
    teardown={},
    ordered=True,
    rel_eps=0.0,
)


def test_write_texts_fall_back_to_the_standard_dialect() -> None:
    assert _WRITE.run_for("ladybug") == "RUN L"
    assert _WRITE.run_for("neug") == "RUN"
    assert _WRITE.check_for("ladybug") == "CHECK"
    assert _WRITE.setup_for("ladybug") == ("S1", "S2")
    assert _WRITE.teardown_for("cypher") == ()
    assert _WRITE.reference_id == "w"
    assert _WRITE.store == "graph"
    assert _WRITE.supported_by(frozenset({"property_graph"}))
    assert not _WRITE.supported_by(frozenset({"sql_pgq"}))


def test_write_reference_holds_before_and_after(tmp_path: Path) -> None:
    doc = catalog.reference_document(_WRITE, [{"ids": ["a"]}], [[["a", 1]]], before=[[]])
    assert doc == {"bindings": [{"ids": ["a"]}], "answers": [[["a", 1]]], "before": [[]]}
    path = tmp_path / "w-small.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert catalog.load_write_reference(path) == ([{"ids": ["a"]}], [[["a", 1]]], [[]])
