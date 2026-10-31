"""On-disk dataset cache keyed by generator and answer sources, scales, bindings, seed and NetworkX version."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import networkx

from graph_bench.datasets import graphalytics, multigraph, rdf, snb
from graph_bench.datasets.schema import Dataset
from graph_bench.suites import catalog

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
    texts = [f"{p.relative_to(_PKG).as_posix()}\n{p.read_text(encoding='utf-8')}" for p in paths]
    # Suite modules added with GRAPH_BENCH_EXTRA_SUITES write references too.
    for module in catalog.EXTRA_SUITES:
        source = Path(module.__file__ or "")
        texts.append(f"{module.__name__}\n{source.read_text(encoding='utf-8')}")
    return texts


def dataset_key(selection: str) -> str:
    h = hashlib.sha256()
    for text in _read_sources():
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    # The BFS, WCC and PageRank references are computed with NetworkX.
    inputs = (selection, catalog.selection(selection).scales, BINDINGS, SEED, networkx.__version__)
    h.update(repr(inputs).encode("utf-8"))
    return h.hexdigest()[:16]


def _write_references(selection: str, snb_ds: Dataset, ga_ds: Dataset, mg_ds: Dataset) -> None:
    """Write `references/<item_id>-<scale>.json` for every reference the selection's items use (`reference_items`).

    Each item's suite module (`catalog.reference_source`) computes them on its dataset's reference graph: the
    `SnbGraph` for SNB, the `Dataset` itself for Graphalytics and the multigraph fixture. A write's file holds the
    rows after the write (`answers`), before it (`before`) and, with `Write.ready`, after its setup (`ready`). An
    item that shares
    another item's references (`Query.same_as`) uses that item's file, written once.
    """
    sel = catalog.selection(selection)
    graphs: dict[str, Any] = {"snb": snb.SnbGraph.from_dataset(snb_ds), "graphalytics": ga_ds, multigraph.NAME: mg_ds}
    for item in catalog.reference_items(selection):
        source = catalog.reference_source(item.id)
        graph = graphs[item.dataset]
        bindings = source.bindings(item.id, graph, sel.bindings)
        results = [source.reference(item.id, graph, b) for b in bindings]
        if isinstance(item, catalog.Write):
            after, before = [r[0] for r in results], [r[1] for r in results]
            ready = [r[2] for r in results] if item.ready else None
            doc = catalog.reference_document(item, bindings, after, before=before, ready=ready)
        else:
            doc = catalog.reference_document(item, bindings, results)
        path = catalog.reference_path(item, snb_ds.scale, base=snb_ds.root.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        mg_root = base / f"{multigraph.NAME}-{scale}"
        roots = (snb_root, ga_root, mg_root)
        if all((root / ".complete").exists() for root in roots):
            snb_ds = Dataset.load(snb_root)
            ga_ds = Dataset.load(ga_root)
            mg_ds = Dataset.load(mg_root)
        else:
            for root in roots:
                if root.exists():
                    shutil.rmtree(root)
            snb_ds = snb.generate(scale, snb_root, SEED)
            ga_ds = graphalytics.generate(snb_ds, ga_root)
            mg_ds = multigraph.generate(scale, mg_root)
            _write_rdf(selection, (snb_ds, ga_ds, mg_ds))
            _write_references(selection, snb_ds, ga_ds, mg_ds)
            for root in roots:
                root.joinpath(".complete").write_text("ok", encoding="utf-8")
        result[("snb", scale)] = snb_ds
        result[("graphalytics", scale)] = ga_ds
        result[(multigraph.NAME, scale)] = mg_ds
    return result
