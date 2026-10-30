"""Hops: multi-hop traversals."""

from benches._cases import make_test
from graph_bench.suites import catalog

for _item in catalog.ITEMS:
    if _item.category == "hops":
        globals()[f"test_{_item.id}"] = make_test(_item)
