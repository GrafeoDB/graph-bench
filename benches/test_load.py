"""Load benchmarks: each dataset bulk-loaded into a fresh instance."""

from benches._cases import make_load_test
from graph_bench.suites import catalog

for _dataset in sorted({_item.dataset for _item in catalog.ITEMS}):
    globals()[f"test_load_{_dataset}"] = make_load_test(_dataset)
