"""Complex queries: the SNB interactive short and complex reads."""

from benches._cases import make_test
from graph_bench.suites import catalog

for _item in catalog.ITEMS:
    if _item.category == "complex_queries":
        globals()[f"test_{_item.id}"] = make_test(_item)
