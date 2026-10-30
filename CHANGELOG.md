# Changelog

## [0.3.0] - Unreleased

graph-bench 0.3 moves the Grafeo adapters to grafeo 0.6 and makes every result say what it measured.

### Added

- **Native, fallback and unsupported labels**: every result records what its timed iterations ran, at run
  time: *native* (every call ran in the database), *fallback (operations)* (part of it ran in Python, such as a
  NetworkX algorithm, a client-side traversal or the vector scan) or *unsupported* (no native implementation
  and no fallback, recorded as SKIPPED). The labels are in the JSON (`metrics.custom`), CSV and Markdown
  reports. See [FAIRNESS.md](FAIRNESS.md#labelling-native-and-fallback-results).
- **Native vector search, `vector_ann_*`**: `vector_ann_build`, `vector_ann_knn` and `vector_ann_delete` measure
  the database's own approximate nearest neighbour index, with recall@k against exact search next to each
  latency, and never fall back. The HNSW parameters are shared by every database in a run (`--ann-m`,
  `--ann-ef-construction`, `--ann-ef`, `--ann-k`), with `--ann-quantization` (Grafeo embedded),
  `--ann-delete-fraction`, `--ann-queries` and a recall versus latency sweep over `ef` (`--ann-sweep`,
  `--ann-sweep-ef`).
- **Both local clustering coefficients**: `ldbc_lcc` is the LDBC Graphalytics directed definition and
  `lcc_undirected` the undirected one. Each adapter lists the definitions it computes natively
  (`native_lcc_definitions`); the other runs the BaseAdapter fallback and is labelled as such.
- **Uni-DB adapter** (`-d unidb`).
- **Tests and CI**: correctness tests for the Grafeo adapters against NetworkX references, benchmark and label
  tests, a CI workflow (lint, ty, tests on Python 3.12 to 3.15) and a weekly run against the newest grafeo on
  PyPI.
- **Python 3.15**: tested on Python 3.15, and the classifiers list 3.12 to 3.15.

### Changed

- **grafeo 0.6**: the `grafeo` extra requires `grafeo>=0.6.0b1,<0.7` (it was `grafeo>=0.5`). grafeo 0.6 has a
  new single-file storage format, copy-on-write checkpoints and planner changes, so results measured with 0.5.x
  are not comparable one to one. See the README section "Grafeo versions".
- **Embedded Grafeo runs more natively**: a BFS or shortest path restricted to one edge type (`snb_ic1`) runs
  Grafeo's algorithm on a projection of that edge type instead of the `get_neighbors` fallback. The adapter loads
  data with the batch APIs, keeps a file database as one `.grafeo` file, and lets query errors fail the
  benchmark instead of timing them.
- **Grafeo Server**: the adapter follows the GWP protocol of grafeo-server 0.5.44 (`CatalogService` in place
  of `DatabaseService`, native search through `AdminService` and `SearchService`), and
  `GRAPH_BENCH_GRAFEO_SERVER_URI` defaults to `localhost:7688`, grafeo-server's GWP port (it was 7687).
- **Vector scan renamed**: `vector_insert`, `vector_knn`, `vector_batch_search` and `vector_recall` are now
  `vector_scan_insert`, `vector_scan_knn`, `vector_scan_batch_search` and `vector_scan_recall`: the same
  brute-force search on every database, which adapters may not override. The old names and `-c vector` still
  select them.
- **Version**: `graph_bench.__version__` matches the package version (it said 0.1.0).
