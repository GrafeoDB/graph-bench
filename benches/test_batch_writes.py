"""Batch writes: write cases, each read back after every run and undone, on their dataset's loaded instance."""

from benches._cases import make_test
from graph_bench.suites import catalog

for _item in catalog.ITEMS:
    if _item.category == "batch_writes":
        globals()[f"test_{_item.id}"] = make_test(_item)
