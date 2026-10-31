"""Pattern matching: cycles, common neighbours, joins and whole-graph 2-hop counts."""

from benches._cases import make_test
from graph_bench.suites import catalog

for _item in catalog.ITEMS:
    if _item.category == "pattern_matching":
        globals()[f"test_{_item.id}"] = make_test(_item)
