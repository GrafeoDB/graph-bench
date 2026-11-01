"""`graph-bench merge`: one results.json from the per-database results of CI's matrix jobs."""

import dataclasses
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graph_bench import merge
from graph_bench.cli import app
from graph_bench.databases.base import DatabaseInfo
from graph_bench.results import Entry, Results
from graph_bench.runner import RUN_DEADLINE_S, supervisor
from graph_bench.runner.supervisor import apply_budget_rule, run_bench
from graph_bench.suites import catalog
from graph_bench.suites.catalog import Case
from graph_bench.types import Timing
from tests.conftest import resources_off

EXTRA = "fake=tests.fakes:FakeDatabase,fake2=tests.fakes:FakeDatabase2"


@pytest.fixture(scope="module", autouse=True)
def _extra_databases() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAPH_BENCH_EXTRA_DATABASES", EXTRA)
        yield


def _id(item_id: str, db: str, scale: str = "small") -> str:
    item = next(i for i in catalog.ITEMS if i.id == item_id)
    return Case(item, db, scale, 1).bench_id


def _entry(db: str, ms: float, *, item: str = "snb_ic13") -> Entry:
    return Entry(
        id=_id(item, db),
        category="hops",
        model="property_graph",
        benchmark=item,
        database=db,
        scale="small",
        threads=1,
        status="ok",
        timing=Timing(ms * 1e6, ms * 1e6, int(ms * 1e6), 5),
        preflight_ns=int(ms * 1e6),
        answer="agree",
        reference="python",
    )


def _doc(*entries: Entry, **fields: object) -> Results:
    info = DatabaseInfo("x", "1", "in-process", False, "", "in-memory")
    base = Results(
        graph_bench_commit="c0ffee",
        grafeo={"version": "0.6.4", "source": "pypi", "sha": None, "git_sha": None},
        machine={"cpu": "Graviton", "cores": 4, "os": "Linux"},
        selection="push",
        mode="bench",
        started_at="2026-11-01T03:00:00Z",
        databases={e.database: dataclasses.replace(info, name=e.database) for e in entries},
        entries=list(entries),
    )
    return dataclasses.replace(base, **fields)  # type: ignore[arg-type]


def _write(tmp_path: Path, name: str, doc: Results) -> Path:
    path = tmp_path / name / "results.json"
    doc.write(path)
    return path


def _verdicts(results: Results) -> dict[str, tuple[str, str, str | None]]:
    """Each entry's status, answer and error, with the ratio of a budget marker dropped (timings vary per run)."""
    return {
        e.id: (e.status, e.answer, re.sub(r"[\d.]+x the top-3", "Nx the top-3", e.error or "") or None)
        for e in results.entries
    }


def test_merge_equals_one_run_over_both_databases(
    push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two single-database runs merged give the entries, statuses, answers and budget markers of one run over both.

    fake2 answers snb_is5 wrongly (≠) and sleeps 0.2 s per graphalytics_wcc call, so the 10x rule marks it ⏱ in
    the two-database run; in its single-database run nothing compares with it, so only the merge can mark it. The
    factor is raised to 100x so that timer noise on the fakes' sub-millisecond cases cannot mark one of them in
    one run and not the other; 0.2 s against fake's well under 1 ms is still beyond it.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    monkeypatch.setenv("GRAPH_BENCH_FAKE", "")
    monkeypatch.setenv("GRAPH_BENCH_FAKE2", "wrong:snb_is5,slow:graphalytics_wcc")
    monkeypatch.setattr(supervisor, "BUDGET_FACTOR", 100)
    resources_off(monkeypatch)
    work = tmp_path / "work"
    both = run_bench("push", ["fake", "fake2"], tmp_path / "both", workdir=work, case_timeout_s=60)
    run_bench("push", ["fake"], tmp_path / "fake", workdir=work, case_timeout_s=60)
    two = run_bench("push", ["fake2"], tmp_path / "fake2", workdir=work, case_timeout_s=60)
    wcc = _id("graphalytics_wcc", "fake2")
    assert {e.status for e in two.entries} <= {"ok", "not_supported"}  # alone, nothing marks it
    assert _verdicts(both)[wcc] == ("timeout", "agree", "too slow: Nx the top-3 average")

    out = tmp_path / "merged"
    # The inputs in the other order: the merged document follows the catalog's order, not the command line's.
    result = CliRunner().invoke(
        app,
        ["merge", "--out", str(out), str(tmp_path / "fake2" / "results.json"), str(tmp_path / "fake" / "results.json")],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip().endswith(str(out / "results.json"))
    merged = Results.read(out / "results.json")
    assert _verdicts(merged) == _verdicts(both)
    assert [e.id for e in merged.entries] == [e.id for e in both.entries]
    assert list(merged.databases) == list(both.databases) == ["fake", "fake2"]
    assert (merged.selection, merged.mode, merged.grafeo) == (both.selection, both.mode, both.grafeo)
    assert merged.started_at == two.started_at  # other top-level fields come from the first input


def test_merge_follows_the_catalog_order(tmp_path: Path) -> None:
    """Databases in the order of `catalog.CI_DATABASES`, then the other built-in ones, then the rest by name;
    entries sorted by id, as one run writes them."""
    paths = [
        _write(tmp_path, "grafeo", _doc(_entry("grafeo", 1))),
        _write(tmp_path, "fake", _doc(_entry("fake", 1))),
        _write(tmp_path, "falkordb", _doc(_entry("falkordb", 1))),
        _write(tmp_path, "aaa", _doc(_entry("aaa", 1))),
        _write(tmp_path, "neug", _doc(_entry("neug", 1))),
    ]
    merged, missing = merge.merge([(path, Results.read(path)) for path in paths])
    assert missing == []
    assert list(merged.databases) == ["neug", "grafeo", "falkordb", "aaa", "fake"]
    assert [e.id for e in merged.entries] == sorted(e.id for e in merged.entries)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection", "full"),
        ("mode", "codspeed"),
        ("graph_bench_commit", "deadbeef"),
        ("grafeo", {"version": "0.6.5", "source": "pypi", "sha": None, "git_sha": None}),
    ],
)
def test_merge_inputs_must_agree(tmp_path: Path, field: str, value: object) -> None:
    """Selection, mode, graph-bench commit and Grafeo source must match; the message names the field and files."""
    first = _write(tmp_path, "a", _doc(_entry("neug", 1)))
    second = _write(tmp_path, "b", _doc(_entry("grafeo", 1), **{field: value}))
    with pytest.raises(merge.MergeError, match=field) as raised:
        merge.merge([(first, Results.read(first)), (second, Results.read(second))])
    assert str(first) in str(raised.value) and str(second) in str(raised.value)
    result = CliRunner().invoke(app, ["merge", "--out", str(tmp_path / "out"), str(first), str(second)])
    assert result.exit_code == 2
    assert field in result.output
    assert not (tmp_path / "out" / "results.json").exists()


def test_merge_rejects_a_duplicate_database(tmp_path: Path) -> None:
    first = _write(tmp_path, "a", _doc(_entry("neug", 1)))
    second = _write(tmp_path, "b", _doc(_entry("neug", 2, item="hop_1")))
    with pytest.raises(merge.MergeError, match="neug") as raised:
        merge.merge([(first, Results.read(first)), (second, Results.read(second))])
    assert str(first) in str(raised.value) and str(second) in str(raised.value)
    result = CliRunner().invoke(app, ["merge", "--out", str(tmp_path / "out"), str(first), str(second)])
    assert result.exit_code == 2


def test_budget_rule_applies_once_at_merge(tmp_path: Path) -> None:
    """A single-database input has no other database to compare with, so its own run marked nothing; the merge
    applies the 10x rule over all entries, which marks the slow one (not twice: a marked entry is no longer ok)."""
    docs = [_doc(_entry("neug", 1)), _doc(_entry("ladybug", 1.2)), _doc(_entry("grafeo", 50))]
    for doc in docs:
        alone = dataclasses.replace(doc, entries=[dataclasses.replace(e) for e in doc.entries])
        apply_budget_rule(alone)
        assert [e.status for e in alone.entries] == ["ok"]
    paths = [_write(tmp_path, doc.entries[0].database, doc) for doc in docs]
    merged, _ = merge.merge([(path, Results.read(path)) for path in paths])
    marked = {e.database: (e.status, e.error) for e in merged.entries}
    assert marked["grafeo"] == ("timeout", "too slow: 45.5x the top-3 average")
    assert marked["neug"] == marked["ladybug"] == ("ok", None)


def test_merge_lists_a_missing_database_as_no_results(tmp_path: Path) -> None:
    """A database of `--db` without an input (its job died) is listed as "no results" and recorded the way a
    database that does not start is: every case it would run ✗ with that error, the rest ⊘, no database info."""
    first = _write(tmp_path, "a", _doc(_entry("fake", 1)))
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["merge", "--out", str(out), "--db", "fake,fake2", str(first)])
    assert result.exit_code == 0, result.output
    assert "no results: fake2" in result.output
    merged = Results.read(out / "results.json")
    assert list(merged.databases) == ["fake"]
    missing = [e for e in merged.entries if e.database == "fake2"]
    sel = catalog.selection("push")
    entries, resources = supervisor.failed_database(sel, "fake2", merge.NO_RESULTS)
    assert sorted(e.id for e in missing) == sorted(e.id for e in entries) == sorted(catalog.all_ids("push", ["fake2"]))
    assert {(e.status, e.error) for e in missing if e.status == "failed"} == {("failed", merge.NO_RESULTS)}
    assert {e.status for e in missing} <= {"failed", "not_supported"}
    assert [r.id for r in merged.resources if r.database == "fake2"] == sorted(r.id for r in resources)
    report = CliRunner().invoke(app, ["report", "--format", "readme", "--in", str(out / "results.json")])
    assert report.exit_code == 0, report.output


def test_ci_settings_prints_the_deadline_and_the_roster() -> None:
    """The workflows read `--deadline-s` and the merge's `--db` from this command, not from YAML literals."""
    result = CliRunner().invoke(app, ["ci-settings"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [f"deadline_s={RUN_DEADLINE_S}", "databases=" + ",".join(catalog.CI_DATABASES)]
