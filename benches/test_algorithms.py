"""Algorithms: the graphalytics algorithms, native implementations only."""

from benches._cases import make_test
from graph_bench.suites import catalog

for _item in catalog.ITEMS:
    if _item.category == "algorithms":
        globals()[f"test_{_item.id}"] = make_test(_item)
