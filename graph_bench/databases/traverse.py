"""Traverse embedded (traverse-embedded): a `.tvdb` file with a redo log, served from memory; Cypher and procedures."""

from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar

import traverse

from graph_bench.databases import PACKAGES
from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets.schema import Dataset
from graph_bench.types import Model

# The load check (`counts`): every node and every relationship, counted with Cypher.
COUNT_NODES = "MATCH (n) RETURN count(n) AS c"
COUNT_EDGES = "MATCH ()-[r]->() RETURN count(r) AS c"
# The engine's own procedures, which project the graph per call (https://github.com/truespar/traverse: RELEASE_NOTES.md
# names `traverse.pageRank.write`, and `traverse-bench` runs `traverse.pageRank.stats` and `traverse.wcc.stats`). The
# GDS-compatible `gds.pageRank` is not used: its scores are on GDS's unnormalized scale
# (docs/reference/apoc-gds-compat.md) and it ignores `scaler: 'L1NORM'`, while `traverse.pageRank` gives the
# probability distribution the reference uses.
PAGE_RANK = (
    "CALL traverse.pageRank.stream({dampingFactor: 0.85, maxIterations: 1000, tolerance: 1e-10}) "
    "YIELD nodeId, score RETURN nodeId, score"
)
WCC = "CALL traverse.wcc.stream({}) YIELD nodeId, communityId RETURN nodeId, communityId"
# One row per vertex reached, the source not included (`algorithm` adds it); totalCost is the hop count.
BFS = "CALL traverse.bfs.stream({sourceNodeId: $source}) YIELD targetNodeId, totalCost RETURN targetNodeId, totalCost"


class TraverseDatabase(Database):
    """A database from `traverse.open`, loaded with the SDK's bulk writer and queried with parameterized Cypher.

    The SDK has no file-free mode: `open` takes a path and logs each commit to a redo log, fsynced, so this adapter is
    labelled on-disk, although queries are served from memory (the package README, "All data is held in memory").
    The redo log costs writes something the in-memory databases do not pay; the SDK has no setting for it.
    """

    name: ClassVar[str] = "traverse"
    display_name: ClassVar[str] = "Traverse"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    dialect: ClassVar[str] = "cypher"

    db: Any = None
    ids: dict[Any, int]
    keys: dict[int, Any]

    @classmethod
    def process_env(cls, threads: int | None) -> dict[str, str]:
        """Parallel query phases run on rayon's global pool, and the embedded SDK builds no pool of its own, so
        rayon's documented variable sets its size (https://docs.rs/rayon/latest/rayon/#global-thread-pool)."""
        return {"RAYON_NUM_THREADS": "1"} if threads == 1 else {}

    def info(self) -> DatabaseInfo:
        note = "RAYON_NUM_THREADS=1" if self.threads == 1 else "rayon pool = all cores"
        return DatabaseInfo(self.name, version(PACKAGES[self.name]), "in-process", True, note, "on-disk")

    def open_empty(self) -> None:
        """An empty database in the workdir: what `load` does before its first node."""
        self.db = traverse.open(str(self.workdir / "bench.tvdb"))

    def store_paths(self) -> list[Path]:
        """The workdir: the `.tvdb` file, its redo log and its lock file, and nothing else of this instance's."""
        return [self.workdir]

    def load(self, dataset: Dataset) -> None:
        """The SDK's bulk path: "committing it saves the database with the load in it, so the data is written once"
        (the package README, "Bulk loading"). The SDK has no CSV loader, so the CSVs are parsed in Python, and
        `traverse-admin import` is not in the wheel and infers each value's type from its text.

        The `id` index of each label is what the `{id: ...}` lookups use. Lookups without a label (IS4 to IS6) have
        no index to use: `create_index` takes a label.
        """
        self.open_empty()
        ids: dict[tuple[str, Any], int] = {}
        with self.db.bulk_writer() as bulk:
            for table in dataset.nodes:
                labels = [table.label]
                for row in dataset.rows(table):
                    ids[table.label, row["id"]] = bulk.create_node(labels, row)
            for edge in dataset.edges:
                for row in dataset.rows(edge):
                    source = ids[edge.src, row.pop("src")]
                    target = ids[edge.dst, row.pop("dst")]
                    bulk.create_edge(source, target, edge.type, row)
            for table in dataset.nodes:
                bulk.create_index(f"{table.label}_id", table.label, "id")
        # The procedures speak internal node ids. Graphalytics has one label, so a dataset id names one node.
        self.ids = {key: nid for (_, key), nid in ids.items()} if dataset.name == "graphalytics" else {}
        self.keys = {nid: key for key, nid in self.ids.items()}

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        # The SDK hands out each row as a dict, keyed by column in column order.
        return [tuple(row.values()) for row in self.db.execute(text, dict(params))]

    def counts(self) -> tuple[int, int]:
        """Nodes and edges loaded, counted with `COUNT_NODES` and `COUNT_EDGES` through `execute()`."""
        return int(self.query(COUNT_NODES, {})[0][0]), int(self.query(COUNT_EDGES, {})[0][0])

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        if name not in ("bfs", "wcc", "pr"):
            raise NotSupported(name)
        if not self.ids:
            raise RuntimeError("the id mapping is kept only for the graphalytics dataset")
        keys = self.keys
        if name == "pr":
            return {keys[nid]: score for nid, score in self.query(PAGE_RANK, {})}
        if name == "wcc":
            return {keys[nid]: component for nid, component in self.query(WCC, {})}
        source = params["source"]
        reached = self.query(BFS, {"source": self.ids[source]})
        return {source: 0, **{keys[nid]: int(cost) for nid, cost in reached}}

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None
