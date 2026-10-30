import re

from typer.testing import CliRunner

from graph_bench.cli import app
from graph_bench.suites import catalog
from graph_bench.suites.catalog import Case, cases
from graph_bench.suites.snb import QUERIES


def _item(item_id: str) -> catalog.Query | catalog.Algorithm:
    return next(i for i in catalog.ITEMS if i.id == item_id)


def test_push_selection_cases() -> None:
    result = cases("push", ["grafeo", "ladybug"])
    assert len(result) == 2 * 13
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


def test_limited_queries_have_total_order() -> None:
    pattern = re.compile(r"(\w+\.)?\w*[iI]d\s+(ASC|DESC)?\s*LIMIT")
    limited = [q for q in QUERIES if "LIMIT" in q.text["cypher"]]
    assert limited
    for q in limited:
        for text in q.text.values():
            assert "ORDER BY" in text, q.id
            assert pattern.search(text), q.id


def test_every_query_has_cypher_text() -> None:
    assert len(QUERIES) == 10
    for q in QUERIES:
        assert q.text["cypher"].strip(), q.id
        assert q.model == "property_graph"
        assert q.dataset == "snb"
