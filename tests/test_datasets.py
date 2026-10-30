import filecmp
import shutil
from pathlib import Path

import networkx
import pytest

from graph_bench.datasets import cache, graphalytics, snb
from graph_bench.datasets.schema import Dataset, EdgeTable, NodeTable


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return snb.generate("small", tmp_path_factory.mktemp("snb") / "small")


def test_generate_is_deterministic(small: Dataset, tmp_path: Path) -> None:
    other = snb.generate("small", tmp_path / "again")
    names = sorted(p.name for p in small.root.glob("*.csv"))
    assert names == sorted(p.name for p in other.root.glob("*.csv"))
    for name in names:
        assert (small.root / name).read_bytes() == (other.root / name).read_bytes(), name
    _, mismatch, errors = filecmp.cmpfiles(small.root, other.root, names, shallow=False)
    assert not mismatch and not errors


def test_schema_roundtrip(small: Dataset) -> None:
    assert Dataset.load(small.root) == small
    person = next(t for t in small.nodes if t.label == "Person")
    row = next(iter(small.rows(person)))
    assert isinstance(row["birthday"], int)
    assert isinstance(row["creationDate"], int)
    assert isinstance(row["id"], str)
    knows = next(t for t in small.edges if t.type == "KNOWS")
    assert isinstance(next(iter(small.rows(knows)))["creationDate"], int)


def _rows(ds: Dataset, tables: tuple[NodeTable, ...] | tuple[EdgeTable, ...]) -> int:
    return sum(1 for t in tables for _ in ds.rows(t))


def test_expected_counts_match_the_csvs(small: Dataset, tmp_path: Path) -> None:
    """The node and edge counts a load is checked against are recorded with the CSVs, in schema.json."""
    assert (small.node_count, small.edge_count) == (_rows(small, small.nodes), _rows(small, small.edges))
    ga = graphalytics.generate(small, tmp_path / "ga")
    assert (ga.node_count, ga.edge_count) == (_rows(ga, ga.nodes), _rows(ga, ga.edges))
    assert Dataset.load(ga.root) == ga


def test_edge_tables_split_by_endpoint_labels(small: Dataset) -> None:
    reply = {(e.src, e.dst) for e in small.edges if e.type == "REPLY_OF"}
    assert reply == {("Comment", "Post"), ("Comment", "Comment")}
    types = {e.type for e in small.edges}
    assert types == set(snb.EDGE_TYPES)


def test_snb_graph_view(small: Dataset) -> None:
    g = snb.SnbGraph.from_dataset(small)
    assert len(g.nodes["Person"]) == 1_001  # 1,000 plus the isolated person
    assert sum(len(v) for v in g.out["KNOWS"].values()) == 18_000
    assert sum(len(v) for v in g.incoming["KNOWS"].values()) == 18_000
    dst, props = (
        g.out["KNOWS"]["person_0"][0] if "person_0" in g.out["KNOWS"] else next(iter(g.out["KNOWS"].values()))[0]
    )
    assert dst.startswith("person_") and isinstance(props["creationDate"], int)


def test_one_isolated_person(small: Dataset) -> None:
    """The last person has no edges at all, so IC13 has an unreachable pair; the rest of the data is unchanged."""
    g = snb.SnbGraph.from_dataset(small)
    touched = {
        node
        for adjacency in (g.out, g.incoming)
        for edges in adjacency.values()
        for node in edges
        if node in g.nodes["Person"]
    }
    assert set(g.nodes["Person"]) - touched == {"person_1000"}


def test_graphalytics_is_symmetric(small: Dataset, tmp_path: Path) -> None:
    ga = graphalytics.generate(small, tmp_path / "ga")
    assert [t.label for t in ga.nodes] == ["V"] and [t.type for t in ga.edges] == ["E"]
    pairs = [(r["src"], r["dst"]) for r in ga.rows(ga.edges[0])]
    assert all(isinstance(a, int) and isinstance(b, int) for a, b in pairs)
    as_set = set(pairs)
    assert len(as_set) == len(pairs)
    assert all((b, a) in as_set for a, b in as_set)
    assert all(a != b for a, b in as_set)
    assert len(list(ga.rows(ga.nodes[0]))) == 1_001


def test_key_changes_with_generator_source(monkeypatch: pytest.MonkeyPatch) -> None:
    before = cache.dataset_key("push")
    original = cache._read_sources()
    monkeypatch.setattr(cache, "_read_sources", lambda: [*original, "# changed"])
    assert cache.dataset_key("push") != before
    assert cache.dataset_key("push") != cache.dataset_key("full")


def test_key_changes_with_answers_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cached references are canonicalized by answers.py and computed with NetworkX: both are in the key."""
    before = cache.dataset_key("push")
    pkg = tmp_path / "graph_bench"
    for sub in ("datasets", "suites"):
        shutil.copytree(cache._PKG / sub, pkg / sub, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(cache._PKG / "answers.py", pkg / "answers.py")
    monkeypatch.setattr(cache, "_PKG", pkg)
    assert cache.dataset_key("push") == before

    with (pkg / "answers.py").open("a", encoding="utf-8") as fh:
        fh.write("\n# canonicalization changed\n")
    changed_answers = cache.dataset_key("push")
    assert changed_answers != before

    monkeypatch.setattr(networkx, "__version__", "0.0.0-test")
    assert cache.dataset_key("push") not in (before, changed_answers)


def test_prepare_reuses_complete_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPH_BENCH_DATA_DIR", str(tmp_path))
    first = cache.prepare("push")
    assert set(first) == {("snb", "small"), ("graphalytics", "small")}
    assert (first[("snb", "small")].root / ".complete").exists()

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("generate called on a complete cache")

    monkeypatch.setattr(snb, "generate", boom)
    monkeypatch.setattr(graphalytics, "generate", boom)
    second = cache.prepare("push")
    assert second == first
