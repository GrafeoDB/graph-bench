"""TuringDB embedded (Linux only): an on-disk graph under the instance's workdir, loaded from Parquet."""

import itertools
import math
import re
import shutil
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar, Literal

from graph_bench.databases.base import Database, DatabaseInfo
from graph_bench.datasets.schema import Dataset
from graph_bench.types import Model

# TuringDB starts a process-wide event handler per engine, and a second engine in one process fails with "Could not
# initialize system event handler" (also after the first client is deleted; the client has no close method). The
# worker loads several times in one process, so the first load starts the engine and every later load of the process
# reuses it, each in a graph of its own.
_engine: dict[str, Any] = {}
_graph_ids = itertools.count()
# The load check (`counts`): a plain count aggregation (https://docs.turingdb.ai/query/cypher_subset lists `count`).
COUNT_NODES = "MATCH (n) RETURN count(n)"
COUNT_EDGES = "MATCH ()-[r]->() RETURN count(r)"

_PARAM = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_ESCAPES = str.maketrans({"\\": "\\\\", "'": "\\'"})


def literal(value: Any) -> str:
    """A Cypher literal for one parameter value.

    TuringDB has no value parameters: `WHERE n.name = $name` is rejected with "Not implemented: Parameters", and
    its documented way is literals in the query text (https://docs.turingdb.ai/query/cypher_subset, "Parameters").
    Strings are single-quoted with the documented backslash escape, so the backslash is escaped first and then the
    quote; ints and floats are numbers; anything else (including bool, None, NaN and infinities) is rejected.
    """
    if isinstance(value, bool):
        raise TypeError(f"cannot inline {type(value).__name__} as a TuringDB literal")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot inline {value!r} as a TuringDB literal")
        return repr(value)
    if isinstance(value, str):
        return "'" + value.translate(_ESCAPES) + "'"
    raise TypeError(f"cannot inline {type(value).__name__} as a TuringDB literal")


def inline(text: str, params: Mapping[str, Any]) -> str:
    """`text` with every `$name` replaced by the `literal` of its value; an unbound name raises KeyError."""
    return _PARAM.sub(lambda m: literal(params[m.group(1)]), text)


def _python(value: Any) -> Any:
    """A DataFrame cell as a Python scalar: numpy scalars to their Python type, missing values to None."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class TuringDatabase(Database):
    """An embedded TuringDB instance (`data_dir` = the workdir) holding one graph loaded with `LOAD PARQUET`.

    TuringDB's fastest documented bulk path is `LOAD PARQUET '<dir>'` (https://docs.turingdb.ai/import_data/parquet):
    `<data_dir>/data/<dir>/nodes.parquet` with `__id` (INT64, unique), `__labels` (list<string>) and one typed column
    per property, and `edges.parquet` with `__source`, `__target` and `__type`. Our CSVs are one file per label, so
    `load()` writes both files with pyarrow (an integer `__id` per node, the application `id` kept as a property)
    and then loads them; the conversion, the instance start and the load are all inside the load clock. Queries are
    the shared openCypher texts with each parameter inlined as a literal; a text TuringDB cannot run fails with
    its own error.
    """

    name: ClassVar[str] = "turingdb"
    display_name: ClassVar[str] = "TuringDB"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    process: ClassVar[Literal["in-process", "local process"]] = "in-process"

    client: Any = None
    graph: str = ""
    staging: Path | None = None

    def info(self) -> DatabaseInfo:
        # The embedded client and the docs expose no thread setting (https://docs.turingdb.ai/), so the engine
        # runs with its own default.
        note = "TuringDB exposes no thread setting"
        # TuringDB has no in-memory mode: `data_dir` holds the graphs (https://docs.turingdb.ai/import_data/parquet).
        return DatabaseInfo(self.name, version("turingdb"), "in-process", False, note, "on-disk")

    def _write_parquet(self, dataset: Dataset, directory: Any) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        directory.mkdir(parents=True, exist_ok=True)
        ids: dict[tuple[str, Any], int] = {}
        node_columns: dict[str, list[Any]] = {"__id": [], "__labels": []}
        property_types: dict[str, str] = {}
        rows_by_label = []
        for table in dataset.nodes:
            for col in table.columns:
                if property_types.setdefault(col.name, col.type) != col.type:
                    raise ValueError(f"property {col.name!r} has different types on different labels")
            rows_by_label.append((table, list(dataset.rows(table))))
        for name in property_types:
            node_columns[name] = []
        count = 0
        for table, rows in rows_by_label:
            for row in rows:
                ids[(table.label, row["id"])] = count
                node_columns["__id"].append(count)
                node_columns["__labels"].append([table.label])
                for name in property_types:
                    node_columns[name].append(row.get(name))
                count += 1
        arrow = {"INT64": pa.int64(), "DOUBLE": pa.float64(), "STRING": pa.string()}
        schema = {"__id": pa.int64(), "__labels": pa.list_(pa.field("element", pa.string()))}
        schema.update({n: arrow[t] for n, t in property_types.items()})
        nodes = pa.table({n: pa.array(v, type=schema[n]) for n, v in node_columns.items()})
        pq.write_table(nodes, directory / "nodes.parquet")

        edge_types: dict[str, str] = {}
        for edge in dataset.edges:
            for col in edge.columns:
                if edge_types.setdefault(col.name, col.type) != col.type:
                    raise ValueError(f"edge property {col.name!r} has different types on different edge types")
        edge_columns: dict[str, list[Any]] = {"__source": [], "__target": [], "__type": []}
        for name in edge_types:
            edge_columns[name] = []
        for edge in dataset.edges:
            for row in dataset.rows(edge):
                edge_columns["__source"].append(ids[(edge.src, row["src"])])
                edge_columns["__target"].append(ids[(edge.dst, row["dst"])])
                edge_columns["__type"].append(edge.type)
                for name in edge_types:
                    edge_columns[name].append(row.get(name))
        edge_schema = {"__source": pa.int64(), "__target": pa.int64(), "__type": pa.string()}
        edge_schema.update({n: arrow[t] for n, t in edge_types.items()})
        edges = pa.table({n: pa.array(v, type=edge_schema[n]) for n, v in edge_columns.items()})
        pq.write_table(edges, directory / "edges.parquet")

    def _start(self) -> Any:
        """The process's engine, started on the first load with its `data_dir` beside the instance's workdir.

        The worker removes each instance's workdir after use, but the engine keeps its `data_dir` for the life of
        the process, so that directory is `<workdir parent>/turingdb` (the same base directory, on the same disk).
        """
        if "client" not in _engine:
            from turingdb import TuringDB

            root = self.workdir.parent / "turingdb"
            root.mkdir(parents=True, exist_ok=True)
            _engine["root"] = root
            _engine["client"] = TuringDB(type="embedded", data_dir=str(root))
        return _engine["client"]

    def load(self, dataset: Dataset) -> None:
        self.client = self._start()
        self.graph = f"bench{next(_graph_ids)}"
        self.staging = _engine["root"] / "data" / self.graph
        self._write_parquet(dataset, self.staging)
        self.client.query(f"LOAD PARQUET '{self.graph}'")
        self.client.set_graph(self.graph)

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        frame = self.client.query(inline(text, params))
        return [tuple(_python(v) for v in row) for row in frame.itertuples(index=False)]

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded, counted with `COUNT_NODES` and `COUNT_EDGES`."""
        return int(self.query(COUNT_NODES, {})[0][0]), int(self.query(COUNT_EDGES, {})[0][0])

    def close(self) -> None:
        # The embedded client has no close method and the engine lives on (see `_engine`); only the Parquet staging
        # files of this load are removed. TuringDB has no DROP GRAPH, so the graph stays in the engine's data_dir.
        self.client = None
        if self.staging is not None:
            shutil.rmtree(self.staging, ignore_errors=True)
            self.staging = None
