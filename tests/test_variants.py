"""A variant is a row beside its database: it never sets another database's budget and is never ranked."""

from graph_bench.databases import KNOWN, MODELS, PACKAGES, REGISTRY, VARIANTS
from graph_bench.report.ranking import ordered_databases, rank
from graph_bench.runner.supervisor import apply_budget_rule
from tests.test_merge import _doc, _entry


def test_the_typed_row_is_a_variant_of_grafeo() -> None:
    assert VARIANTS == {"grafeo_typed": "grafeo"}
    assert "grafeo_typed" in KNOWN
    assert "grafeo_typed" in REGISTRY
    assert PACKAGES["grafeo_typed"] == "grafeo"
    assert MODELS["grafeo_typed"] == frozenset({"property_graph"})


def test_a_variant_sets_no_budget_for_another_database() -> None:
    """A fast variant would otherwise pull the top-3 average down and mark `slow`."""
    doc = _doc(_entry("grafeo_typed", 1), _entry("neug", 100), _entry("ladybug", 100), _entry("slow", 150))
    apply_budget_rule(doc)
    assert {e.database: e.status for e in doc.entries} == {
        "grafeo_typed": "ok",
        "neug": "ok",
        "ladybug": "ok",
        "slow": "ok",
    }


def test_a_variant_has_a_budget_of_its_own() -> None:
    doc = _doc(_entry("grafeo_typed", 5000), _entry("grafeo", 100), _entry("neug", 100))
    apply_budget_rule(doc)
    marked = {e.database: e.status for e in doc.entries}
    assert marked == {"grafeo_typed": "timeout", "grafeo": "ok", "neug": "ok"}


def test_a_variant_is_never_ranked() -> None:
    """The headline stays the schemaless row: the variant neither takes a rank nor pushes Grafeo down."""
    ranked = rank(_doc(_entry("grafeo_typed", 1), _entry("grafeo", 2), _entry("neug", 3)))[0]
    assert [db for db, _ in ranked.order] == ["grafeo", "neug"]
    assert ranked.position("grafeo") == 1
    assert ranked.position("grafeo_typed") is None
    assert ranked.fastest() == {"grafeo"}


def test_a_variant_sorts_right_after_its_database() -> None:
    names = ["neug", "grafeo_typed", "duckdb", "grafeo"]
    assert ordered_databases(names) == ["grafeo", "grafeo_typed", "duckdb", "neug"]
