"""RDF/SPARQL: SNB queries in SPARQL over the RDF mapping of the dataset."""

from benches._cases import make_test
from graph_bench.suites import catalog

for _item in catalog.ITEMS:
    if _item.category == "rdf":
        globals()[f"test_{_item.id}"] = make_test(_item)
