"""Grafeo adapter: the push selection agrees with the references, except the recorded known disagreements."""

import re
from pathlib import Path

import pytest

pytest.importorskip("grafeo")

from graph_bench.databases import get_database
from graph_bench.runner.supervisor import run_bench

pytestmark = pytest.mark.grafeo

KNOWN_DISAGREEMENTS = Path(__file__).resolve().parent.parent / "docs" / "known-disagreements.md"


def _known(db: str) -> set[str]:
    """Benchmark ids recorded for `db` in docs/known-disagreements.md, from its `- id: <db>/<benchmark>` lines."""
    if not KNOWN_DISAGREEMENTS.exists():
        return set()
    text = KNOWN_DISAGREEMENTS.read_text(encoding="utf-8")
    return {m.group(2) for m in re.finditer(r"^- id: (\S+)/(\S+)\s*$", text, re.MULTILINE) if m.group(1) == db}


def test_grafeo_push_agrees(push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    results = run_bench("push", ["grafeo"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
    known = _known("grafeo")
    entries = [e for e in results.entries if e.database == "grafeo"]
    assert entries
    failing = []
    for entry in entries:
        if entry.benchmark in known or entry.status == "not_supported":
            continue
        good = entry.status == "ok" and (entry.category == "load" or entry.answer == "agree")
        if not good:
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    print("\n".join(failing))
    assert not failing, "\n".join(failing)


def test_grafeo_single_thread_env() -> None:
    cls = get_database("grafeo")
    assert cls.process_env(1) == {"RAYON_NUM_THREADS": "1"}
    assert cls.process_env(None) == {}
