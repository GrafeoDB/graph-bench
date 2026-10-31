import dataclasses
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graph_bench.cli import app
from graph_bench.databases.base import DatabaseInfo
from graph_bench.report.ranking import rank
from graph_bench.report.readme import render_readme, update_readme
from graph_bench.report.summary import render_summary
from graph_bench.results import Entry, Results
from graph_bench.types import AnswerVerdict, Category, Status, Timing

LEGEND = "⊘ not available · ✗ failing · ≠ wrong answer · ⏱ beyond 10x the top-3 average"

EXPECTED_SUMMARY = """\
<!-- graph-bench:start -->
**graph-bench** · Grafeo 0.6.1 (abcdef1) · Test CPU, 8 cores, TestOS 1 · 2026-10-30 · 3 embedded databases

| Category | DBs | Grafeo fastest | Grafeo median rank |
|---|---|---|---|
| Complex queries | 2 | 1 / 1 | 1 |
| Algorithms | 2 | 1 / 2 | 1.5 |

Single-threaded, small scale; answers checked; wrong answers not ranked.

Grafeo: ≠ on snb_ic6

⊘ not available · ✗ failing · ≠ wrong answer · ⏱ beyond 10x the top-3 average

Full results, with every database named: https://github.com/GrafeoDB/graph-bench#results
<!-- graph-bench:end -->
"""


def _entry(
    category: Category,
    benchmark: str,
    db: str,
    status: Status,
    answer: AnswerVerdict = "agree",
    ms: float | None = None,
    *,
    scale: str = "small",
    threads: int | None = 1,
    load_s: float | None = None,
) -> Entry:
    timing = Timing(ms * 1e6, ms * 1e6, int(ms * 1e6), 5) if ms is not None else None
    return Entry(
        id=f"benches/test_{category}.py::test_{benchmark}[{db}-{scale}]",
        category=category,
        model="property_graph",
        benchmark=benchmark,
        database=db,
        scale=scale,
        threads=threads,
        status=status,
        timing=timing,
        preflight_ns=int(ms * 1e6) if ms is not None else None,
        load_ns=int(load_s * 1e9) if load_s is not None else None,
        answer=answer,
    )


def _info(name: str) -> DatabaseInfo:
    return DatabaseInfo(name, "1.0", "in-process", True, f"{name} threads note", "in-memory")


@pytest.fixture
def results() -> Results:
    """2 categories, 4 benchmarks, 3 databases; every marker appears at least once."""
    cq: Category = "complex_queries"
    algo: Category = "algorithms"
    entries = [
        _entry(cq, "snb_is1", "grafeo", "ok", "agree", 0.395),
        _entry(cq, "snb_is1", "ladybug", "ok", "agree", 1.2),
        _entry(cq, "snb_is1", "fake", "failed", "unchecked"),
        # Grafeo is fastest here but disagrees, so it must not be ranked.
        _entry(cq, "snb_ic6", "grafeo", "ok", "disagree", 10.0),
        _entry(cq, "snb_ic6", "ladybug", "ok", "agree", 45.151),
        # Too slow: the timing is kept but the entry is never ranked.
        _entry(cq, "snb_ic6", "fake", "timeout", "agree", 497.032),
        _entry(algo, "graphalytics_bfs", "grafeo", "ok", "agree", 2.0),
        _entry(algo, "graphalytics_bfs", "ladybug", "ok", "agree", 1.5),
        _entry(algo, "graphalytics_bfs", "fake", "not_supported", "unchecked"),
        _entry(algo, "graphalytics_pr", "grafeo", "ok", "agree", 3.0),
        _entry(algo, "graphalytics_pr", "ladybug", "ok", "agree", 30.0),
        _entry(algo, "graphalytics_pr", "fake", "not_supported", "unchecked"),
    ]
    return Results(
        graph_bench_commit="0123456789abcdef",
        grafeo={"version": "0.6.1", "source": "wheel", "sha": "abcdef1234567890"},
        machine={"cpu": "Test CPU", "cores": 8, "os": "TestOS 1"},
        selection="push",
        mode="bench",
        started_at="2026-10-30T16:04:34Z",
        databases={name: _info(name) for name in ("grafeo", "ladybug", "fake")},
        entries=entries,
    )


def _row(text: str, label: str) -> list[str]:
    """The cells of the first table row whose first cell is `label`."""
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] == label:
            return cells[1:]
    raise AssertionError(f"no row {label!r} in:\n{text}")


def test_summary_exact_output(results: Results) -> None:
    assert render_summary(results) == EXPECTED_SUMMARY


def test_summary_names_no_other_database(results: Results) -> None:
    text = render_summary(results)
    assert "LadybugDB" not in text
    assert "ladybug" not in text.lower()
    assert "fake" not in text.lower()
    assert "3 embedded databases" in text


def test_summary_lists_grafeo_markers(results: Results) -> None:
    lines = render_summary(results).splitlines()
    assert "Grafeo: ≠ on snb_ic6" in lines
    assert LEGEND in lines


def test_summary_without_grafeo_markers_has_no_legend(results: Results) -> None:
    results.entries = [e for e in results.entries if not (e.database == "grafeo" and e.benchmark == "snb_ic6")]
    text = render_summary(results)
    assert "Grafeo:" not in text
    assert LEGEND not in text


def test_summary_excludes_disagree_from_ranking(results: Results) -> None:
    text = render_summary(results)
    # Grafeo's 10 ms on snb_ic6 disagrees: it is not counted, so Grafeo is fastest on 1 of 1, not 2 of 2.
    assert _row(text, "Complex queries") == ["2", "1 / 1", "1"]
    ic6 = next(r for r in rank(results) if r.benchmark == "snb_ic6")
    assert [db for db, _ in ic6.order] == ["ladybug"]


def test_summary_counts_grafeo_timeouts_that_agree(results: Results) -> None:
    """n counts every benchmark where Grafeo's answer agreed, ⏱ included; k still counts first places only."""
    results.entries += [
        _entry("algorithms", "graphalytics_wcc", "grafeo", "timeout", "agree", 300.0),
        _entry("algorithms", "graphalytics_wcc", "ladybug", "ok", "agree", 3.0),
        # A ⏱ killed before its answer was checked does not count.
        _entry("algorithms", "graphalytics_cdlp", "grafeo", "timeout", "unchecked"),
        _entry("algorithms", "graphalytics_cdlp", "ladybug", "ok", "agree", 3.0),
    ]
    text = render_summary(results)
    assert _row(text, "Algorithms") == ["2", "1 / 3", "1.5"]
    assert "Grafeo: ⏱ on graphalytics_wcc" in text.splitlines()


def test_summary_has_no_combined_score(results: Results) -> None:
    for line in render_summary(results).lower().splitlines():
        assert "overall" not in line
        assert "total score" not in line


def test_summary_uses_largest_scale_single_threaded(results: Results) -> None:
    results.entries += [
        _entry("hops", "snb_ic13", "grafeo", "ok", "agree", 5.0, scale="medium"),
        _entry("hops", "snb_ic13", "ladybug", "ok", "agree", 4.0, scale="medium"),
        _entry("hops", "snb_ic13", "grafeo", "ok", "agree", 1.0, scale="medium", threads=None),
    ]
    text = render_summary(results)
    assert "Single-threaded, medium scale;" in text
    assert _row(text, "Hops") == ["2", "0 / 1", "2"]
    assert "Complex queries" not in text


def test_summary_scale_follows_grafeo(results: Results) -> None:
    # Another database ran at a larger scale than Grafeo: the summary stays at Grafeo's largest scale.
    results.entries.append(_entry("hops", "snb_ic13", "ladybug", "ok", "agree", 4.0, scale="large"))
    text = render_summary(results)
    assert "Single-threaded, small scale;" in text
    assert _row(text, "Complex queries") == ["2", "1 / 1", "1"]
    assert "Hops" not in text


def test_summary_footer_lines_are_separate_paragraphs(results: Results) -> None:
    lines = render_summary(results).splitlines()
    footer = lines[lines.index("| Algorithms | 2 | 1 / 2 | 1.5 |") + 1 : -1]
    assert footer[0::2] == ["", "", "", ""]
    assert all(footer[1::2])
    assert len(footer) == 8


def test_summary_label_keeps_dirty_marker(results: Results) -> None:
    results.grafeo["sha"] = "abcdef1234567890-dirty"
    assert "Grafeo 0.6.1 (abcdef1-dirty)" in render_summary(results)


def test_summary_n_skips_failed_entries_that_agreed(results: Results) -> None:
    """An entry whose preflight agreed but whose timed runs then failed (✗) is not in n."""
    results.entries += [
        _entry("algorithms", "graphalytics_wcc", "grafeo", "failed", "agree", 3.0),
        _entry("algorithms", "graphalytics_wcc", "ladybug", "ok", "agree", 3.0),
    ]
    assert _row(render_summary(results), "Algorithms") == ["2", "1 / 2", "1.5"]


def test_summary_n_counts_loads_unless_their_counts_disagree(results: Results) -> None:
    results.entries += [
        _entry("load", "load_snb", "grafeo", "ok", "agree", load_s=0.131),
        _entry("load", "load_snb", "ladybug", "ok", "unchecked", load_s=0.165),
        _entry("load", "load_graphalytics", "grafeo", "ok", "disagree", load_s=0.01),
        _entry("load", "load_graphalytics", "ladybug", "ok", "agree", load_s=0.165),
    ]
    assert _row(render_summary(results), "Load") == ["2", "1 / 1", "1"]


def test_summary_load_ranks_by_load_time(results: Results) -> None:
    results.entries += [
        _entry("load", "load_snb", "grafeo", "ok", "unchecked", load_s=0.131),
        _entry("load", "load_snb", "ladybug", "ok", "unchecked", load_s=0.165),
    ]
    assert _row(render_summary(results), "Load") == ["2", "1 / 1", "1"]


def test_summary_requires_grafeo(results: Results) -> None:
    results.entries = [e for e in results.entries if e.database != "grafeo"]
    del results.databases["grafeo"]
    with pytest.raises(ValueError, match="Grafeo"):
        render_summary(results)


def test_rank_codspeed_mode_uses_preflight(results: Results) -> None:
    results.mode = "codspeed"
    for e in results.entries:
        e.timing = None
    results.entries[1].preflight_ns = 100  # ladybug snb_is1 becomes the fastest
    is1 = next(r for r in rank(results) if r.benchmark == "snb_is1")
    assert [db for db, _ in is1.order] == ["ladybug", "grafeo"]
    assert is1.title == "IS1 person profile"


def test_readme_columns_grafeo_first_then_alphabetical(results: Results) -> None:
    text = render_readme(results)
    assert _row(text, "Category") == ["Grafeo", "fake", "LadybugDB"]


def test_readme_overview_counts(results: Results) -> None:
    text = render_readme(results)
    # Columns: Grafeo, fake, LadybugDB.
    assert _row(text, "Complex queries") == ["✓ 1/2 ≠1", "✓ 0/2 ✗1 ⏱1", "✓ 2/2"]
    assert _row(text, "Algorithms") == ["✓ 2/2", "⊘", "✓ 2/2"]


def test_readme_detail_markers(results: Results) -> None:
    text = render_readme(results)
    assert _row(text, "IS1 person profile") == ["**0.395**", "✗", "1.20"]
    assert _row(text, "IC6 tag co-occurrence") == ["≠", "⏱", "**45.2**"]
    assert _row(text, "BFS") == ["2.00", "⊘", "**1.50**"]
    assert _row(text, "PageRank") == ["**3.00**", "⊘", "30.0"]
    assert "### Complex queries (small scale, ms)" in text
    assert LEGEND in text


def test_readme_footer_metadata(results: Results) -> None:
    text = render_readme(results)
    assert "graph-bench 0123456" in text
    assert "Grafeo 0.6.1 (abcdef1)" in text
    assert "Test CPU, 8 cores, TestOS 1" in text
    assert _row(text, "LadybugDB") == ["1.0", "in-process", "in-memory", "ladybug threads note"]
    assert "overall" not in text.lower()


def test_readme_load_table_in_seconds(results: Results) -> None:
    results.entries += [
        _entry("load", "load_snb", "grafeo", "ok", "unchecked", load_s=0.131),
        _entry("load", "load_snb", "ladybug", "ok", "unchecked", load_s=1.6466),
        _entry("load", "load_snb", "fake", "failed", "unchecked"),
    ]
    text = render_readme(results)
    assert "### Load (small scale, s)" in text
    assert _row(text, "LDBC SNB") == ["**0.131**", "✗", "1.65"]
    assert _row(text, "Load") == ["✓ 1/1", "✓ 0/1 ✗1", "✓ 1/1"]


def test_readme_labels_scales_when_several(results: Results) -> None:
    results.entries += [dataclasses.replace(e, scale="medium") for e in results.entries]
    text = render_readme(results)
    assert _row(text, "Complex queries, small") == ["✓ 1/2 ≠1", "✓ 0/2 ✗1 ⏱1", "✓ 2/2"]
    assert "### Algorithms (medium scale, ms)" in text


def test_update_readme_replaces_between_markers(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# Title\n\n<!-- results:start -->\nold\n<!-- results:end -->\n\nTail\n", encoding="utf-8")
    update_readme(path, "new body\n")
    assert path.read_text(encoding="utf-8") == (
        "# Title\n\n<!-- results:start -->\nnew body\n<!-- results:end -->\n\nTail\n"
    )


def test_update_readme_requires_markers(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# Title\n\n<!-- results:start -->\nno end\n", encoding="utf-8")
    with pytest.raises(ValueError, match="results:end"):
        update_readme(path, "body")
    assert path.read_text(encoding="utf-8") == "# Title\n\n<!-- results:start -->\nno end\n"


def test_cli_report_summary(results: Results, tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    results.write(path)
    out = CliRunner().invoke(app, ["report", "--format", "summary", "--in", str(path)])
    assert out.exit_code == 0, out.output
    assert out.output == EXPECTED_SUMMARY


def test_cli_report_readme_write(results: Results, tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    results.write(path)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n", encoding="utf-8")
    out = CliRunner().invoke(app, ["report", "--format", "readme", "--in", str(path), "--write", str(readme)])
    assert out.exit_code == 0, out.output
    assert "IS1 person profile" in readme.read_text(encoding="utf-8")


def test_cli_report_rejects_bad_usage(results: Results, tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    results.write(path)
    runner = CliRunner()
    write = ["--write", str(tmp_path / "README.md")]
    assert runner.invoke(app, ["report", "--format", "summary", "--in", str(path), *write]).exit_code == 2
    assert runner.invoke(app, ["report", "--format", "html", "--in", str(path)]).exit_code == 2
    assert runner.invoke(app, ["report", "--format", "summary", "--in", str(tmp_path / "none.json")]).exit_code == 2
    missing = runner.invoke(app, ["report", "--format", "readme", "--in", str(path), *write])
    assert missing.exit_code == 2


def test_load_rows_name_the_rdf_store_and_follow_the_graph_loads() -> None:
    from graph_bench.report import ranking

    assert ranking.title("load_snb") == "LDBC SNB"
    assert ranking.title("load_snb_rdf") == "LDBC SNB (RDF)"
    order = sorted(["load_snb_rdf", "load_graphalytics", "load_snb"], key=ranking._benchmark_key)
    assert order == ["load_snb", "load_graphalytics", "load_snb_rdf"]


def test_a_database_without_an_rdf_store_shows_not_available_in_the_load_row(results: Results) -> None:
    """The worker reports a load a database's models do not reach as `not_supported`, so the Load table shows ⊘
    for it and the overview counts it as ⊘, not as a missing correct load."""
    load: Category = "load"
    results.entries += [
        _entry(load, "load_snb", "grafeo", "ok", "agree", load_s=0.137),
        _entry(load, "load_snb", "ladybug", "ok", "agree", load_s=0.136),
        _entry(load, "load_snb_rdf", "grafeo", "ok", "agree", load_s=0.621),
        _entry(load, "load_snb_rdf", "ladybug", "not_supported", "unchecked"),
    ]
    text = render_readme(results)
    assert _row(text, "Category")[:3] == ["Grafeo", "fake", "LadybugDB"]
    assert _row(text, "LDBC SNB (RDF)") == ["**0.621**", "", "⊘"]
    assert _row(text, "LDBC SNB") == ["0.137", "", "**0.136**"]
    assert _row(text, "Load") == ["✓ 2/2", "", "✓ 1/2 ⊘1"]
