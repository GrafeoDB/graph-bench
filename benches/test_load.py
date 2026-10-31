"""Load benchmarks: each dataset bulk-loaded into a fresh instance, once per store its items read."""

from benches._cases import load_keys, make_load_test
from graph_bench.suites import catalog

for _dataset, _store in load_keys():
    globals()[f"test_{catalog.load_benchmark(_dataset, _store)}"] = make_load_test(_dataset, _store)
