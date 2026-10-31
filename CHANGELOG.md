# Changelog

## [0.3.0] - Unreleased

graph-bench 0.3 is rebuilt as a per-process, answer-checked comparison of embedded graph databases. The 0.2.x
benchmark moves to `legacy/` untouched and is deleted once the new comparison replaces all of it.

### Added

- **Embedded comparison**: ten embedded databases, all implemented: Grafeo, LadybugDB, NeuG, FalkorDB (through
  falkordblite), TuringDB, HelixDB, ArcadeDB, Neo4j Community, Oxigraph (RDF/SPARQL only) and DuckDB + DuckPGQ
  (SQL/PGQ only), on Load (SNB, Graphalytics), Complex queries (SNB IS1 to IS7, IC2, IC6), Hops (IC13) and
  Algorithms (Graphalytics BFS, WCC, PageRank). See the README for the categories and their status. The JVM
  databases share one heap cap and start their JVM outside every clock (`graph_bench/databases/jvm.py`).
- **Data models**: a database declares one dialect per model it serves, and the harness routes each query by
  its model (`Database.dialects`, `query_model`). SQL/PGQ (SNB IS1, IS4, IS6, IC13 in standard SQL/PGQ) and
  RDF/SPARQL (SNB IS1, IS4, IS6 in SPARQL over a shared RDF mapping of the CSVs) reuse the SNB references.
  Property graph and SQL/PGQ read one loaded graph; RDF has its own load and Load row (`load_snb_rdf`), from an
  N-Triples file written when the data is prepared. Algorithms run on any database with a property graph or
  SQL/PGQ model; a load a database's models do not reach is ⊘. Grafeo runs all three models.
- **Grafeo build provenance**: `setup --grafeo <wheel> --grafeo-sha <commit>` records the grafeo commit a wheel was
  built from as `git_sha` in `results.json`, beside the wheel's SHA-256; a checkout records its own commit. The
  summary label shows the commit.
- **One process per database**: `graph-bench bench` runs one worker process per database, one after another, and
  writes `results.json` (schema in the design spec) with the machine, the Grafeo build and each database's
  version, process kind, storage and thread setting.
- **Answer check and preflight**: every query's answer is compared with a Python reference computed from the
  generated data (NetworkX for algorithms, the LDBC epsilon for PageRank). Each benchmark first runs once,
  untimed, where the answer check and the time budget apply; only a benchmark that passes is timed.
- **Time budget**: 10x the average median of the 3 fastest other databases with correct answers, applied over
  all results at the end, so the run order never decides a marker from this final rule. During the run a
  runaway guard cuts a preflight's database call (cold first run, harness work excluded) off at 30x the average
  of the 3 fastest earlier preflights and never below 10 s, once 3 earlier results exist; this guard depends on which databases ran
  earlier. Before that a hard cap applies (`--case-timeout`, default 600 s of worker silence). Startup and both
  load rows are exempt from the budget.
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
  selection in wall-time mode on CodSpeed's macro runners against the newest stable release of every database (grafeo from PyPI too), and a daily
  `full` run that opens a pull request updating the README results. `pins.yml` checks weekly for a newer Neo4j
  Community and Temurin JRE and opens a pull request on `bot/pins`.
- **Startup row**: every database's cold start, timed in fresh child processes (`graph_bench/runner/startup.py`):
  the driver's import, an empty database opened (`Database.open_empty`, starting whatever the process lacks) and
  one probe query checked against the empty-database answer; downloads and installs run before, untimed
  (`Database.provision`). The median over 3 processes in `push`, 5 in `full`; in CodSpeed, the whole child
  processes are timed. Adapters implement `open_empty` in a later step; until then their Startup entry is ⊘.
- **Load (query language) row**: each graph-store dataset loaded through the database's query language
  (`Database.load_ql`) into fresh instances that no query uses, count-checked like Load plus a content probe,
  run after the dataset's queries; a failure fails only that entry. Adapters implement `load_ql` in a later step.
- **Write cases**: the harness for Batch writes (`suites.Write`): untimed setup, a timed committed write per batch,
  untimed read-back and teardown after every run, a check of the loaded state before the first run and a node
  and edge count guard after the case; write cases run after every read query of their dataset.
- **Selections**: `push` now plans Startup, Load (query language), Batch writes, Scans and Pattern matching too
  (their queries arrive with the next steps), and both selections turn the Memory and Disk records on.
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
