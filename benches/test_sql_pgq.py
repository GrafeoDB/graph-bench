"""SQL/PGQ: SNB queries in SQL/PGQ (ISO/IEC 9075-16) over the property graph."""

from benches._cases import make_test
from graph_bench.suites import catalog

for _item in catalog.ITEMS:
    if _item.category == "sql_pgq":
        globals()[f"test_{_item.id}"] = make_test(_item)
