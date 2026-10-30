"""Dataset schema: tables, columns, and typed CSV reading."""

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ColumnType = Literal["INT64", "DOUBLE", "STRING"]

_CASTS = {"INT64": int, "DOUBLE": float, "STRING": str}


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType


@dataclass(frozen=True)
class NodeTable:
    """A node table. The first column is `id`, the primary key."""

    label: str
    columns: tuple[Column, ...]
    file: str


@dataclass(frozen=True)
class EdgeTable:
    """An edge table. CSV header is `src,dst,<columns>`."""

    type: str
    src: str
    dst: str
    columns: tuple[Column, ...]
    file: str


def _col_to_json(c: Column) -> dict[str, str]:
    return {"name": c.name, "type": c.type}


def _cols_from_json(items: list[dict[str, str]]) -> tuple[Column, ...]:
    return tuple(Column(i["name"], i["type"]) for i in items)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Dataset:
    name: str
    scale: str
    root: Path
    nodes: tuple[NodeTable, ...]
    edges: tuple[EdgeTable, ...]

    def save(self) -> None:
        doc = {
            "name": self.name,
            "scale": self.scale,
            "nodes": [
                {"label": n.label, "file": n.file, "columns": [_col_to_json(c) for c in n.columns]} for n in self.nodes
            ],
            "edges": [
                {
                    "type": e.type,
                    "src": e.src,
                    "dst": e.dst,
                    "file": e.file,
                    "columns": [_col_to_json(c) for c in e.columns],
                }
                for e in self.edges
            ],
        }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "schema.json").write_text(json.dumps(doc, indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, root: Path) -> "Dataset":
        doc = json.loads((root / "schema.json").read_text(encoding="utf-8"))
        nodes = tuple(NodeTable(n["label"], _cols_from_json(n["columns"]), n["file"]) for n in doc["nodes"])
        edges = tuple(
            EdgeTable(e["type"], e["src"], e["dst"], _cols_from_json(e["columns"]), e["file"]) for e in doc["edges"]
        )
        return cls(doc["name"], doc["scale"], root, nodes, edges)

    def _id_cast(self, label: str) -> Any:
        for n in self.nodes:
            if n.label == label:
                return _CASTS[n.columns[0].type]
        return str

    def rows(self, table: NodeTable | EdgeTable) -> Iterator[dict[str, Any]]:
        """Yield each CSV row as a dict with values typed per column."""
        casts: dict[str, Any] = {c.name: _CASTS[c.type] for c in table.columns}
        if isinstance(table, EdgeTable):
            casts = {"src": self._id_cast(table.src), "dst": self._id_cast(table.dst), **casts}
        with (self.root / table.file).open(encoding="utf-8", newline="") as fh:
            for raw in csv.DictReader(fh):
                yield {k: casts[k](v) for k, v in raw.items()}
