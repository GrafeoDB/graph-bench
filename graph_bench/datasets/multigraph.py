"""The multigraph fixture: a tiny fixed graph with the graphalytics schema, V(id) and E(V->V), whose E rows hold
self-loops and parallel edges, the cases the generated datasets never have.

It is the data of the trail-semantics case (`sem_trail_self_loop`, C1 of the analytics brief): the 2-hop text
`MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN count(*)` gives a different count when an engine matches walks (an edge
may repeat in a match) instead of trails (openCypher relationship isomorphism: no edge twice), and on this graph
only a self-loop makes them differ. Every property-graph adapter loads it as it loads graphalytics; an adapter that
rejects a self-loop or merges parallel rows fails or disagrees on that case, which is its marker.

Its rows, by intent:

- 1->2 twice: parallel edges, two relationships between the same ordered pair;
- 2->1: the antiparallel edge of that pair;
- 2->2, 5->5: a self-loop on a vertex with other edges;
- 3->3 twice: parallel self-loops (a trail may take one after the other, but not one twice);
- 2->3, 3->4, 4->5, 5->4: a chain and another antiparallel pair;
- vertex 6 has no edge.

Trails: the sum over vertices of in-degree times out-degree (26) minus one per self-loop (4) is 22; an engine that
matches walks counts 26. The data does not depend on the scale; a copy is written per scale like the other datasets.
"""

import csv
from pathlib import Path

from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable

NAME = "multigraph"

VERTICES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
EDGES: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 3),
    (3, 3),
    (3, 4),
    (4, 5),
    (5, 4),
    (5, 5),
)


def generate(scale: str, root: Path) -> Dataset:
    """Write the fixture's CSVs and schema.json into `root`, labelled with `scale`."""
    root.mkdir(parents=True, exist_ok=True)
    node = NodeTable("V", (Column("id", "INT64"),), "V.csv")
    edge = EdgeTable("E", "V", "V", (), "E.csv")
    with (root / node.file).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id"])
        w.writerows([v] for v in VERTICES)
    with (root / edge.file).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["src", "dst"])
        w.writerows(EDGES)
    ds = Dataset(NAME, scale, root, (node,), (edge,), len(VERTICES), len(EDGES))
    ds.save()
    return ds
