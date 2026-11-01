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
from graph_bench.results import Entry, Resource, Results
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


def test_merge_write_failure_exits_2(tmp_path: Path) -> None:
    """An output that cannot be written fails like an unreadable input: exit 2 and a message, no traceback."""
    first = _write(tmp_path, "a", _doc(_entry("neug", 1)))
    blocker = tmp_path / "out"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    result = CliRunner().invoke(app, ["merge", "--out", str(blocker), str(first)])
    assert result.exit_code == 2, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "merge: " in result.output and "results.json" in result.output


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


LOCAL_MACHINE = {"cpu": "Ryzen 9", "cores": 16, "os": "Linux 6.6 (Docker)"}


def _local(*entries: Entry) -> Results:
    """A local snapshot: another machine, another day, another graph-bench commit and Grafeo."""
    return _doc(
        *entries,
        machine=LOCAL_MACHINE,
        started_at="2026-10-20T10:00:00Z",
        graph_bench_commit="10ca1",
        grafeo={"version": "0.6.3", "source": "pypi", "sha": None, "git_sha": None},
    )


def _resource(db: str) -> Resource:
    return Resource(
        id=f"resources::load_snb[{db}-small]",
        database=db,
        dataset="snb",
        store="graph",
        scale="small",
        threads=1,
        status="ok",
        footprint_bytes=1 << 20,
    )


def test_merge_local_adds_the_snapshot_databases_not_in_ci(tmp_path: Path) -> None:
    """`--local`: the snapshot's databases that no CI input holds join the merge, every entry, record and info
    tagged `local`; its other databases are dropped (CI measured them); `sources` keeps both runs."""
    ci = [_write(tmp_path, "neug", _doc(_entry("neug", 1))), _write(tmp_path, "grafeo", _doc(_entry("grafeo", 2)))]
    snapshot = _local(_entry("neug", 9), _entry("grafeo", 9), _entry("falkordb", 3), _entry("helixdb", 4))
    snapshot.resources = [_resource("falkordb"), _resource("neug")]
    local = _write(tmp_path, "local", snapshot)
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["merge", "--out", str(out), "--local", str(local), *map(str, ci)])
    assert result.exit_code == 0, result.output
    assert "local: falkordb" in result.output and "local: helixdb" in result.output
    assert "local: neug" not in result.output
    merged = Results.read(out / "results.json")
    assert list(merged.databases) == ["neug", "grafeo", "falkordb", "helixdb"]
    sources = {name: info.source for name, info in merged.databases.items()}
    assert sources == {"neug": "ci", "grafeo": "ci", "falkordb": "local", "helixdb": "local"}
    assert {(e.database, e.source, e.preflight_ns) for e in merged.entries} == {
        ("neug", "ci", 1_000_000),
        ("grafeo", "ci", 2_000_000),
        ("falkordb", "local", 3_000_000),
        ("helixdb", "local", 4_000_000),
    }
    assert [(r.database, r.source) for r in merged.resources] == [("falkordb", "local")]
    # The top level stays the CI run's; each source's date and machine are in `sources`.
    assert (merged.machine, merged.started_at) == (_doc().machine, _doc().started_at)
    assert merged.run("ci")["machine"] == _doc().machine
    assert merged.run("local")["machine"] == LOCAL_MACHINE
    assert merged.run("local")["started_at"] == "2026-10-20T10:00:00Z"
    assert merged.run("local")["graph_bench_commit"] == "10ca1"


def test_merge_without_local_has_no_sources(tmp_path: Path) -> None:
    path = _write(tmp_path, "neug", _doc(_entry("neug", 1)))
    merged, _ = merge.merge([(path, Results.read(path))])
    assert merged.sources == {}
    assert {e.source for e in merged.entries} == {"ci"}


@pytest.mark.parametrize(("field", "value"), [("selection", "full"), ("mode", "codspeed")])
def test_merge_local_must_measure_the_same_way(tmp_path: Path, field: str, value: object) -> None:
    """The snapshot may come from another machine, day, commit and Grafeo, not another selection or mode."""
    ci = _write(tmp_path, "ci", _doc(_entry("neug", 1)))
    local = _write(tmp_path, "local", dataclasses.replace(_local(_entry("falkordb", 1)), **{field: value}))
    with pytest.raises(merge.MergeError, match=field):
        merge.merge([(ci, Results.read(ci))], local=(local, Results.read(local)))
    result = CliRunner().invoke(app, ["merge", "--out", str(tmp_path / "out"), "--local", str(local), str(ci)])
    assert result.exit_code == 2
    assert field in result.output


def test_merge_local_never_fills_a_missing_ci_database(tmp_path: Path) -> None:
    """A roster database whose CI job left no results stays ✗ "no results": a local timing never stands in for a
    CI one."""
    ci = _write(tmp_path, "ci", _doc(_entry("fake", 1)))
    local = _write(tmp_path, "local", _local(_entry("fake2", 1), _entry("falkordb", 1)))
    merged, missing = merge.merge([(ci, Results.read(ci))], ["fake", "fake2"], local=(local, Results.read(local)))
    assert missing == ["fake2"]
    fake2 = [e for e in merged.entries if e.database == "fake2"]
    assert fake2 and {e.source for e in fake2} == {"ci"}
    assert {e.status for e in fake2} <= {"failed", "not_supported"}
    assert merged.databases["falkordb"].source == "local"


def test_merge_local_never_takes_a_roster_database_without_db(tmp_path: Path) -> None:
    """Without `--db` nothing is "missing", yet a roster database (`catalog.CI_DATABASES`) is still never taken
    from the snapshot: it belongs to CI, with or without a CI input."""
    ci = _write(tmp_path, "ci", _doc(_entry("grafeo", 1)))
    snapshot = _local(_entry("neug", 1), _entry("ladybug", 1), _entry("falkordb", 1))
    local = _write(tmp_path, "local", snapshot)
    merged, missing = merge.merge([(ci, Results.read(ci))], local=(local, Results.read(local)))
    assert missing == []
    assert list(merged.databases) == ["grafeo", "falkordb"]
    assert {e.database for e in merged.entries} == {"grafeo", "falkordb"}
    assert [db for db in catalog.CI_DATABASES if merge.from_snapshot(db)] == []
    assert merge.from_snapshot("falkordb")


def test_merge_keeps_the_local_verdicts(tmp_path: Path) -> None:
    """The final budget rule runs over the CI entries only. The snapshot's own run budgeted its entries over all
    ten databases; once its roster databases are dropped, fewer than three local peers remain, and a slow roster
    peer no longer lifts the mean. The merge keeps the snapshot's verdict, never a new ⏱."""
    snapshot = _local(_entry("neug", 50), _entry("falkordb", 1), _entry("helixdb", 2), _entry("neo4j", 20))
    apply_budget_rule(snapshot)  # the snapshot's own run: neo4j's budget is 10x mean(1, 2, 50)
    assert {e.status for e in snapshot.entries} == {"ok"}
    local = _write(tmp_path, "local", snapshot)
    ci = _write(tmp_path, "ci", _doc(_entry("grafeo", 1)))
    merged, _ = merge.merge([(ci, Results.read(ci))], local=(local, Results.read(local)))
    assert {e.database: (e.status, e.error) for e in merged.entries} == {
        "grafeo": ("ok", None),
        "falkordb": ("ok", None),
        "helixdb": ("ok", None),
        "neo4j": ("ok", None),  # 13x mean(1, 2) over the local peers alone
    }


def test_budget_rule_never_compares_sources() -> None:
    """The 10x rule runs within each source: a very fast local timing sets no budget for a CI entry, and a CI
    timing none for a local one; within a source the rule still applies."""
    entries = [
        _entry("neug", 1),
        _entry("ladybug", 1.2),
        _entry("grafeo", 5),  # 4.5x the CI top-3: within budget, though 5000x the local falkordb
        dataclasses.replace(_entry("falkordb", 0.001), source="local"),
        dataclasses.replace(_entry("helixdb", 0.002), source="local"),
        dataclasses.replace(_entry("neo4j", 0.5), source="local"),  # 333x the local top-3: marked
        dataclasses.replace(_entry("oxigraph", 0.0001, item="hop_1"), source="local"),
        _entry("duckdb", 3, item="hop_1"),  # alone in CI on hop_1: no budget, whatever the local timing
    ]
    doc = _doc(*entries)
    apply_budget_rule(doc)
    status = {e.database: e.status for e in doc.entries}
    assert status == {
        "neug": "ok",
        "ladybug": "ok",
        "grafeo": "ok",
        "falkordb": "ok",
        "helixdb": "ok",
        "neo4j": "timeout",
        "oxigraph": "ok",
        "duckdb": "ok",
    }
