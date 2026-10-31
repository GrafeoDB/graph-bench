"""Load (query language): each graph-store dataset loaded through the database's query language, on fresh
instances that no case uses."""

from benches._cases import load_ql_datasets, make_load_ql_test
from graph_bench.suites import catalog

for _dataset in load_ql_datasets():
    globals()[f"test_{catalog.load_ql_benchmark(_dataset)}"] = make_load_ql_test(_dataset)
