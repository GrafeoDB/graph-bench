# Changelog

## [0.3.0] - Unreleased

graph-bench 0.3 is rebuilt as a per-process, answer-checked comparison of embedded graph databases. The 0.2.x
benchmark moves to `legacy/` untouched and is deleted once the new comparison replaces all of it.

### Added

- **Embedded comparison**: ten embedded databases are planned (Grafeo, LadybugDB, TuringDB, FalkorDB through
  falkordblite, NeuG, ArcadeDB, Neo4j Community, DuckDB + DuckPGQ, Oxigraph, HelixDB). Grafeo, LadybugDB, NeuG and
  FalkorDB are implemented, on Load (SNB, Graphalytics), Complex queries (SNB IS1 to IS7, IC2, IC6), Hops (IC13) and
  Algorithms (Graphalytics BFS, WCC, PageRank). See the README for the categories and their status.
- **Data models**: a database declares one dialect per model it serves, and the harness routes each query by
  its model (`Database.dialects`, `query_model`). SQL/PGQ (SNB IS1, IS4, IS6, IC13 in standard SQL/PGQ) and
  RDF/SPARQL (SNB IS1, IS4, IS6 in SPARQL over a shared RDF mapping of the CSVs) reuse the SNB references.
  Property graph and SQL/PGQ read one loaded graph; RDF has its own load and Load row (`load_snb_rdf`). Grafeo
  runs all three models.
- **One process per database**: `graph-bench bench` runs one worker process per database, one after another, and
  writes `results.json` (schema in the design spec) with the machine, the Grafeo build and each database's
  version, process kind, storage and thread setting.
- **Answer check and preflight**: every query's answer is compared with a Python reference computed from the
  generated data (NetworkX for algorithms, the LDBC epsilon for PageRank). Each benchmark first runs once,
  untimed, where the answer check and the time budget apply; only a benchmark that passes is timed.
- **Time budget**: 10x the average median of the 3 fastest other databases with correct answers, applied over
  all results at the end, so the run order never decides a marker from this final rule. During the run a
  runaway guard cuts a preflight's database call (cold first run, harness work excluded) off at 30x the average
  of the 3 fastest earlier preflights, once 3 earlier results exist; this guard depends on which databases ran
  earlier. Before that a hard cap applies (`--case-timeout`, default 600 s of worker silence). Loads are exempt
  from the budget.
- **Load count check**: right after each timed load, outside the clock, the adapter's optional `counts()` counts
  the nodes and edges loaded through the database's own query interface; a mismatch with the dataset's counts
  (recorded in schema.json when the CSVs are written) makes the load ≠. Grafeo and LadybugDB count with Cypher.
- **Markers**: ⊘ not available, ✗ failing, ≠ wrong answer, ⏱ beyond the budget, each mapped to a fixed pytest
  outcome and reason prefix for CodSpeed.
- **Datasets**: an SNB-schema generator and the Graphalytics graph (the SNB `KNOWS` graph in both directions),
  seeded, written as typed CSV files with their reference answers into `GRAPH_BENCH_DATA_DIR` (default
  `~/.cache/graph-bench`). `graph-bench data prepare` fills the cache, `graph-bench data key` prints its key.
- **Commands**: `setup --grafeo` (install a grafeo wheel, or build one from a checkout with maturin, and record
  version and sha), `list`, `bench`, `codspeed` (one `pytest benches --codspeed` process per database) and
  `report --format summary|readme` (the summary block for grafeo's README, or the matrix for this README,
  written between its results markers with `--write`).
- **Selections**: `push` (small scale, the core property graph categories) and `full` (every category and
  scale), both single-threaded.
- **CI**: `ci.yml` (lint and tests on Python 3.12 and 3.13, including the adapter agreement tests) and
  `bench.yml`, graph-bench's own CodSpeed project, separate from grafeo's CodSpeed benchmarks: the `push`
  selection in wall-time mode on CodSpeed's macro runners against grafeo `main` built from source, and a daily
  `full` run that opens a pull request updating the README results.
- **Known disagreements**: `docs/known-disagreements.md` records wrong answers, slow cases and engine findings as
  they are, instead of working around them in the adapters.

### Changed

- **Fairness rules**: FAIRNESS.md is rewritten for the embedded comparison: one process per database,
  single-threaded runs (all-cores runs return with grafeo 0.6.4), in-memory storage where supported, each
  database's fastest documented bulk load, its vendor-recommended API, checked answers and the budget rule.
- **README and CONTRIBUTING**: rewritten for the new comparison and for adding a database adapter.
- **Dependencies**: the extras are `grafeo`, `ladybug`, `embedded` (both), `bench` (pytest-codspeed) and `dev`.

### Removed

- **The 0.2.x benchmark** and its unreleased 0.3 work (native and fallback labels, `vector_ann_*`, the Uni-DB
  adapter, the grafeo-server adapter) move to [legacy/](legacy/README.md), together with the server databases
  and the Docker setup. Results measured with it are not comparable with the new comparison.
