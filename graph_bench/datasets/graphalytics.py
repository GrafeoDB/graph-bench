"""Graphalytics-style graph derived from SNB KNOWS: V(id) and symmetric E(V->V)."""

import csv
from pathlib import Path

from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable


def _num(person_id: str) -> int:
    return int(person_id.rsplit("_", 1)[1])


def generate(snb: Dataset, root: Path) -> Dataset:
    person = next(t for t in snb.nodes if t.label == "Person")
    knows = next(t for t in snb.edges if t.type == "KNOWS")
    vertices = sorted(_num(r["id"]) for r in snb.rows(person))
    pairs: set[tuple[int, int]] = set()
    for r in snb.rows(knows):
        a, b = _num(r["src"]), _num(r["dst"])
        if a != b:
            pairs.add((a, b))
            pairs.add((b, a))
    root.mkdir(parents=True, exist_ok=True)
    node = NodeTable("V", (Column("id", "INT64"),), "V.csv")
    edge = EdgeTable("E", "V", "V", (), "E.csv")
    with (root / node.file).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id"])
        w.writerows([v] for v in vertices)
    with (root / edge.file).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["src", "dst"])
        w.writerows(sorted(pairs))
    ds = Dataset("graphalytics", snb.scale, root, (node,), (edge,))
    ds.save()
    return ds
