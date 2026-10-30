"""On-disk dataset cache keyed by generator sources, scales, bindings and seed."""

import hashlib
import os
import shutil
from pathlib import Path

from graph_bench.datasets import graphalytics, snb
from graph_bench.datasets.schema import Dataset

SEED = 42
# Dataset-to-adapter loading contract; bump when the CSV layout the adapters consume changes.
BINDINGS = "csv-v1"

# Task 4's selection catalog becomes the source of these; it replaces this table.
_SELECTION_SCALES: dict[str, tuple[str, ...]] = {"push": ("small",), "full": ("small", "medium")}

_PKG = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    override = os.environ.get("GRAPH_BENCH_DATA_DIR")
    return Path(override) if override else Path.home() / ".cache" / "graph-bench"


def _read_sources() -> list[str]:
    """Texts of the generator sources (datasets and suites); a missing directory contributes nothing."""
    texts = []
    for sub in ("datasets", "suites"):
        for path in sorted((_PKG / sub).glob("*.py")):
            texts.append(f"{sub}/{path.name}\n{path.read_text(encoding='utf-8')}")
    return texts


def dataset_key(selection: str) -> str:
    h = hashlib.sha256()
    for text in _read_sources():
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    h.update(repr((selection, _SELECTION_SCALES[selection], BINDINGS, SEED)).encode("utf-8"))
    return h.hexdigest()[:16]


def prepare(selection: str) -> dict[tuple[str, str], Dataset]:
    """Generate (or reuse) the datasets for a selection, keyed by (dataset name, scale)."""
    base = data_dir() / dataset_key(selection)
    result: dict[tuple[str, str], Dataset] = {}
    for scale in _SELECTION_SCALES[selection]:
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
            snb_ds.root.joinpath(".complete").write_text("ok", encoding="utf-8")
            ga_ds.root.joinpath(".complete").write_text("ok", encoding="utf-8")
        result[("snb", scale)] = snb_ds
        result[("graphalytics", scale)] = ga_ds
    return result
