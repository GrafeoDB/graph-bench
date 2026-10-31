"""On-disk dataset cache keyed by generator and answer sources, scales, bindings, seed and NetworkX version."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import networkx

from graph_bench.datasets import graphalytics, rdf, snb
from graph_bench.datasets.schema import Dataset
from graph_bench.suites import catalog
from graph_bench.suites import graphalytics as ga_suite
from graph_bench.suites import snb as snb_suite

SEED = 42
# Dataset-to-adapter loading contract; bump when the CSV layout or the N-Triples file the adapters consume changes.
BINDINGS = "csv-v1,nt-v1"

_PKG = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    override = os.environ.get("GRAPH_BENCH_DATA_DIR")
    return Path(override) if override else Path.home() / ".cache" / "graph-bench"


def _read_sources() -> list[str]:
    """Texts of the sources the cached files depend on.

    The generators (datasets/, suites/) and answers.py, which canonicalizes the stored references.
    A missing directory contributes nothing.
    """
    paths = [p for sub in ("datasets", "suites") for p in sorted((_PKG / sub).glob("*.py"))]
    paths.append(_PKG / "answers.py")
    # The pinned downloads (Neo4j tarball, Temurin JRE) are cached under the same directory (`~/.cache/graph-bench`),
    # so a changed pin must change the key too. Read as text: the module has no driver imports.
    paths.append(_PKG / "databases" / "pins.py")
    return [f"{p.relative_to(_PKG).as_posix()}\n{p.read_text(encoding='utf-8')}" for p in paths]


def dataset_key(selection: str) -> str:
    h = hashlib.sha256()
    for text in _read_sources():
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    # The BFS, WCC and PageRank references are computed with NetworkX.
    inputs = (selection, catalog.selection(selection).scales, BINDINGS, SEED, networkx.__version__)
    h.update(repr(inputs).encode("utf-8"))
    return h.hexdigest()[:16]


def _write_references(selection: str, snb_ds: Dataset, ga_ds: Dataset) -> None:
    """Write `references/<item_id>-<scale>.json` for every reference the selection's items use (`reference_items`).

    An item that shares another item's references (`Query.same_as`) uses that item's file, written once.
    """
    sel = catalog.selection(selection)
    graph = snb.SnbGraph.from_dataset(snb_ds)
    for item in catalog.reference_items(selection):
        if isinstance(item, catalog.Query):
            bindings = snb_suite.bindings(item.id, graph, sel.bindings)
            answers: list[Any] = [snb_suite.reference(item.id, graph, b) for b in bindings]
        else:
            bindings = ga_suite.bindings(item.id, ga_ds)
            answers = [ga_suite.reference(item.id, ga_ds, b) for b in bindings]
        path = catalog.reference_path(item, snb_ds.scale, base=snb_ds.root.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = catalog.reference_document(item, bindings, answers)
        path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8", newline="\n")


def _write_rdf(selection: str, datasets: tuple[Dataset, ...]) -> None:
    """Write the N-Triples file (`rdf.write_ntriples`) of every dataset an RDF item of the selection reads."""
    wanted = {item.dataset for item in catalog.items(selection) if item.store == "rdf"}
    for ds in datasets:
        if ds.name in wanted:
            rdf.write_ntriples(ds)


def prepare(selection: str) -> dict[tuple[str, str], Dataset]:
    """Generate (or reuse) the datasets and reference answers for a selection, keyed by (dataset name, scale)."""
    base = data_dir() / dataset_key(selection)
    result: dict[tuple[str, str], Dataset] = {}
    for scale in catalog.selection(selection).scales:
        snb_root = base / f"snb-{scale}"
        ga_root = base / f"graphalytics-{scale}"
        if (snb_root / ".complete").exists() and (ga_root / ".complete").exists():
            snb_ds = Dataset.load(snb_root)
            ga_ds = Dataset.load(ga_root)
        else:
            for root in (snb_root, ga_root):
                if root.exists():
                    shutil.rmtree(root)
            snb_ds = snb.generate(scale, snb_root, SEED)
            ga_ds = graphalytics.generate(snb_ds, ga_root)
            _write_rdf(selection, (snb_ds, ga_ds))
            _write_references(selection, snb_ds, ga_ds)
            snb_ds.root.joinpath(".complete").write_text("ok", encoding="utf-8")
            ga_ds.root.joinpath(".complete").write_text("ok", encoding="utf-8")
        result[("snb", scale)] = snb_ds
        result[("graphalytics", scale)] = ga_ds
    return result
