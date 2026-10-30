"""Fake databases for the runner tests. They answer from the reference files, so they agree.

Behaviour flags, a comma list in `GRAPH_BENCH_FAKE` (FakeDatabase) or `GRAPH_BENCH_FAKE2` (FakeDatabase2):
`crash:<item_id>` exits the process with code 3, `hang:<item_id>` sleeps an hour, `slow:<item_id>` sleeps
0.2 s per call, `wrong:<item_id>` returns a perturbed answer and `unsupported:<item_id>` raises NotSupported.
"""

import os
import time
from collections.abc import Mapping
from typing import Any, ClassVar

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets.schema import Dataset
from graph_bench.suites import Algorithm, Query, catalog
from graph_bench.types import Model


class FakeDatabase(Database):
    name: ClassVar[str] = "fake"
    display_name: ClassVar[str] = "Fake"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    env_var: ClassVar[str] = "GRAPH_BENCH_FAKE"

    dataset: Dataset | None = None

    def info(self) -> DatabaseInfo:
        return DatabaseInfo(self.name, "1.0", "in-process", False, "fake, no threads", "in-memory")

    def load(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def _flags(self, item_id: str) -> set[str]:
        flags = set()
        for part in os.environ.get(self.env_var, "").split(","):
            flag, _, target = part.strip().partition(":")
            if target == item_id:
                flags.add(flag)
        return flags

    def _answer(self, item: Query | Algorithm, params: Mapping[str, Any]) -> Any:
        assert self.dataset is not None, "query before load"
        flags = self._flags(item.id)
        if "crash" in flags:
            os._exit(3)
        if "hang" in flags:
            time.sleep(3600)
        if "slow" in flags:
            time.sleep(0.2)
        if "unsupported" in flags:
            raise NotSupported(f"{self.name} does not run {item.id}")
        path = catalog.reference_path(item, self.dataset.scale, base=self.dataset.root.parent)
        bindings, answers = catalog.load_reference(path)
        answer = answers[bindings.index(dict(params))]
        if "wrong" in flags:
            if isinstance(item, Algorithm):
                return {**answer, -1: 0}
            return [*answer, ["wrong"]]
        return answer

    def _item(self, match: Any) -> Query | Algorithm:
        assert self.dataset is not None, "query before load"
        name = self.dataset.name
        return next(i for i in catalog.ITEMS if i.dataset == name and match(i))

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        item = self._item(lambda i: isinstance(i, Query) and i.text_for(self.dialect) == text)
        return [tuple(row) for row in self._answer(item, params)]

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        item = self._item(lambda i: isinstance(i, Algorithm) and i.name == name)
        return dict(self._answer(item, params))


class FakeDatabase2(FakeDatabase):
    name: ClassVar[str] = "fake2"
    display_name: ClassVar[str] = "Fake 2"
    env_var: ClassVar[str] = "GRAPH_BENCH_FAKE2"


class FakeRdfDatabase(FakeDatabase):
    """Supports only RDF, so every property-graph case is not available."""

    name: ClassVar[str] = "fake_rdf"
    display_name: ClassVar[str] = "Fake RDF"
    models: ClassVar[frozenset[Model]] = frozenset({"rdf"})
