"""TuringDB adapter (embedded, Linux only): the push selection agrees except the recorded known disagreements."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

from graph_bench.databases import get_database
from graph_bench.databases.base import NotSupported
from graph_bench.databases.turingdb import inline, literal
from graph_bench.datasets import cache
from graph_bench.runner.supervisor import run_bench
from graph_bench.types import Model
from tests.conftest import not_yet_implemented, resources_off, warn_if_fixed

# The worker's model gate error, exactly: f"model {case.item.model} not supported".
MODEL_GATE = re.compile(f"model ({'|'.join(get_args(Model))}) not supported")

BS = "\\"
Q = "'"


def test_literal_quotes_and_backslashes() -> None:
    assert literal("plain") == Q + "plain" + Q
    assert literal("it's") == Q + "it" + BS + Q + "s" + Q
    assert literal("a" + BS + "b") == Q + "a" + BS * 2 + "b" + Q
    # A backslash before a quote must not let the quote close the string: the pair BS+Q becomes BS BS BS Q.
    assert literal(BS + Q) == Q + BS * 3 + Q + Q
    assert literal("x' OR 1=1 //") == Q + "x" + BS + "' OR 1=1 //" + Q
    assert literal("") == "''"


def test_literal_unicode_and_newline_pass_through() -> None:
    assert literal("Zoë 日本 \U0001f600") == "'Zoë 日本 \U0001f600'"
    assert literal("a\nb") == "'a\nb'"


def test_literal_numbers_and_rejections() -> None:
    assert literal(42) == "42"
    assert literal(-7) == "-7"
    assert literal(2.5) == "2.5"
    for bad in (True, None, [1], {"a": 1}, float("nan"), float("inf"), b"x"):
        with pytest.raises((TypeError, ValueError)):
            literal(bad)


def test_inline_replaces_every_parameter() -> None:
    text = "MATCH (n {id: $a}) WHERE n.x <> $a AND n.y = $b RETURN n"
    quoted = Q + "p" + BS + "'1" + Q
    expected = f"MATCH (n {{id: {quoted}}}) WHERE n.x <> {quoted} AND n.y = 3 RETURN n"
    assert inline(text, {"a": "p'1", "b": 3}) == expected
    # A value that looks like a parameter is not substituted again.
    assert inline("RETURN $a", {"a": "$b", "b": 1}) == "RETURN '$b'"
    with pytest.raises(KeyError):
        inline("RETURN $missing", {})


@pytest.mark.turingdb
def test_turingdb_push_agrees(
    push_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    known_disagreements: Callable[[str], set[str]],
) -> None:
    pytest.importorskip("turingdb")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    resources_off(monkeypatch)  # Task 4 turns them on with test_<db>_resources_ok
    results = run_bench("push", ["turingdb"], tmp_path / "out", workdir=tmp_path / "work", case_timeout_s=60)
    known = known_disagreements("turingdb")
    entries = [e for e in results.entries if e.database == "turingdb"]
    assert entries
    failing = []
    for entry in entries:
        # ⊘ passes only for the worker's model gate; any other ⊘ (a missing capability) must be allowlisted in
        # docs/known-disagreements.md.
        lacks_model = entry.status == "not_supported" and MODEL_GATE.fullmatch(entry.error or "") is not None
        warn_if_fixed(entry, known)
        if entry.benchmark in known or lacks_model or not_yet_implemented(entry):
            continue
        good = entry.status == "ok" and entry.answer == "agree"
        if not good:
            failing.append(f"{entry.id}: status={entry.status} answer={entry.answer} error={entry.error}")
            failing.append(f"    detail={entry.detail}")
    print("\n".join(failing))
    assert not failing, "\n".join(failing)


@pytest.mark.turingdb
def test_turingdb_info_and_algorithms(push_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("turingdb")
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(push_data_dir))
    cls = get_database("turingdb")
    info = cls(tmp_path, 1).info()
    assert info.process == "in-process"
    assert info.storage == "on-disk"
    assert not info.threads_configurable
    assert "thread" in info.threads_note
    db = cls(tmp_path, 1)
    try:
        dataset = cache.prepare("push")[("graphalytics", "small")]
        db.load(dataset)
        assert db.counts() == (dataset.node_count, dataset.edge_count)
        for name in ("bfs", "wcc", "pr", "cdlp"):
            with pytest.raises(NotSupported):
                db.algorithm(name, {"source": 0})
    finally:
        db.close()
