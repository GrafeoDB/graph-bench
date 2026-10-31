"""Scans: label scans, property filters, an aggregation and an order plus limit, in openCypher, SQL/PGQ and SPARQL."""

from benches._cases import make_test
from graph_bench.suites import catalog

for _item in catalog.ITEMS:
    if _item.category == "scans":
        globals()[f"test_{_item.id}"] = make_test(_item)
