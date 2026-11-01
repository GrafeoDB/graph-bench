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
  summary label shows the commit. The record counts only while the installed grafeo is that build (its
  `direct_url.json`, when present, names the recorded wheel); a wheel installed over it is reported by its local
  path or URL.
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
  earlier. Before that only the run-time limits below and the silence limit apply (`--case-timeout`, default
  120 s of worker silence). Startup and both load rows are exempt from the budget.
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
- **Scans**: label scan, property filter, aggregation, order plus limit and property filter with average over the
  SNB comments and persons (`graph_bench/suites/scans.py`), each with an SQL/PGQ and a SPARQL twin in the same
  category that shares its references, so DuckDB + DuckPGQ and Oxigraph take part. HelixDB runs four through DSL
  builders (the aggregation is ⊘). The order plus limit scan is the Load (query language) content probe.
- **Hops**: 1-hop, 2-hop and variable length 1 to 3 over KNOWS in either direction, each a distinct count of the
  persons reached (`graph_bench/suites/hops.py`), beside IC13. Explicit node-distinctness conditions make
  relationship-isomorphic and walk-based engines give the same answer, and the references check that KNOWS pairs are
  unique and loop-free. Each has an SQL/PGQ twin and, but for the variable length (SPARQL 1.1 has no bounded
  repetition), a SPARQL twin in the same category. HelixDB runs all three through DSL builders.
- **Pattern matching**: triangles and 4-cycles through a person, common friends and a multi-pattern join on SNB, each
  with an SQL/PGQ and a SPARQL twin; triangles through a hub person and over the whole graph; and the analytics
  brief's whole-graph 2-hop count, 2-hop count per source (top 10) and out-degree (top 10) on Graphalytics
  (`graph_bench/suites/patterns.py`). Explicit node-distinctness conditions make relationship-isomorphic and
  walk-based engines agree on the SNB patterns. The trail-semantics case runs the 2-hop count on a new multigraph
  fixture with self-loops and parallel edges (`graph_bench/datasets/multigraph.py`), where trails and walks differ;
  it is answer-only (`catalog.ANSWER_ONLY_DATASETS`): no budget, no ranking, ✓ or its marker in the matrix, and the
  fixture has no Load or Memory and Disk rows. HelixDB runs common friends and the 2-hop count through DSL builders.
- **Memory and Disk records**: after each database's worker (or pytest process), the supervisor loads every
  dataset and store the database's models reach once more, each in a fresh child process
  (`graph_bench/runner/resources.py`), and samples the child's whole process tree every 10 ms from outside
  (`graph_bench/runner/sampler.py`, psutil RSS; a child subreaper on Linux keeps daemonized servers in the tree).
  Each record holds the resident memory before and after the load, the footprint (after minus before), the peak
  during the load and, after `close()`, the size of the store's files (`Database.store_paths`; "in memory" for an
  in-memory store), with the load's count check. They go to the new `resources` list of `results.json` (schema 2,
  which also records `machine.memory_bytes`; schema 1 files still read), never to CodSpeed, and to the README's
  Memory and Disk tables. The startup harness now uses the same subreaper, which ends and reaps the processes it
  adopted when it exits.
- **Known disagreements**: `docs/known-disagreements.md` records wrong answers, slow cases and engine findings as
  they are, instead of working around them in the adapters.
- **One macro-runner job per roster database**: bench.yml's `codspeed` and `readme` jobs and release.yml's `post` job
  are matrices over `catalog.CI_DATABASES` (NeuG, LadybugDB, TuringDB, Oxigraph, DuckDB + DuckPGQ, Grafeo), one leg at
  a time (`max-parallel: 1`, `fail-fast: false`, 55-minute timeout), each on a fresh box. Each leg passes what is left
  of the run deadline (`graph-bench ci-settings`, counted from the job's first step, never below 0) as `--deadline-s`
  and the results of the legs before it in the roster as `--yardstick` (picked by the new `graph-bench ci-yardsticks`,
  so a re-run or out-of-order leg never takes a later one), a new option of `bench` and `codspeed` whose correct
  entries set the in-run budgets as earlier databases of the same run would. A bench worker gets the deadline too
  (`--deadline-at`) and starts no case or load past it; `--deadline-total-s`, the whole 48 minutes, is the figure the
  ⏱ message names ("run deadline: 48 min"). Every leg installs one grafeo version, resolved once per run
  (bench.yml's `resolve` job, release.yml's `detect`; the prepare action's new `grafeo-version` input).
  `codspeed-results`, `readme-results` and `post-results` on ubuntu-latest merge the legs, unless the run was
  cancelled or the legs were skipped, with the new `graph-bench merge` (inputs must agree on selection, mode,
  graph-bench commit and Grafeo; the 10x rule applies once over the merged CI entries; a leg without results shows as
  ✗ "no results"), then upload the merged results, open the README pull request or post to Discord as before. A
  release run whose merged results hold no Grafeo posts and records nothing, with a warning. Every upload overwrites,
  so a re-run job can upload again. The `macro-runner` concurrency group moves to workflow level on both workflows
  (replacing release.yml's `release-post`), since a job-level group would let matrix legs cancel each other.
- **Local snapshot of all ten** (FAIRNESS.md, "The CI roster and the local snapshot"): `results/local/results.json`,
  written by `.github/scripts/local-bench.sh` (the full selection over every database in one `ubuntu:24.04` container on
  the dev box) and committed. `graph-bench merge --local FILE` adds its databases outside the CI roster.
  results.json schema 3 tags every entry, Memory and Disk record and database info with its `source` (`ci` or `local`)
  and records each source's run (date, machine, commit, Grafeo) in `sources`; an older document reads as `ci`. The 10x
  rule and every rank (README bold, the summary block's counts) stay within one source. The README puts the CI columns
  first and the local ones after, marked `†` with a footnote naming the snapshot's date and machine; the summary block
  counts them, marked, and both link the new overview page. `graph-bench report --format overview --write
  docs/results.md` writes that page: every database's version, source, machine and run date, then every per-case table.
  `graph-bench local-check --local FILE` lists the snapshot's databases outside the roster whose newest stable release
  on PyPI (Neo4j: its pin) differs from the version it measured. bench.yml's `readme-results` merges the snapshot when
  it is committed, writes `docs/results.md` into the pull request and raises a notice from `local-check`. The merge jobs
  download the legs into `legs/`, since `results/*/results.json` would match the snapshot.
- **Run-time limits** (FAIRNESS.md rule 9), the same for every database at any position in the run order, so a
  run fits in the CodSpeed macro runner's hour: every case stops 60 s after it starts (with at least 3 timed runs
  it keeps its timing, else ⏱ "too slow: over 60 s"; the supervisor kills a single call still running 5 s past
  the ceiling); the load ceiling applies per Load row (one dataset in one store): a load still running 5 minutes
  after it starts is stopped, ⏱ "load over 5 min", together with the cases that read that load, the dataset's
  Load (query language) row and the load's Memory and Disk record; and a case starts no further timed run
  once its timed runs total 5 s and at least 3 ran (loads: 1). The recorded run count is what ran. In
  `graph-bench codspeed` the box goes to pytest-codspeed's `--codspeed-max-time`, with `--codspeed-max-rounds` at
  the selection's iterations so that fast cases keep their run count there too. The macro-runner jobs
  (bench.yml `codspeed` and `readme`, release.yml `post`) get `timeout-minutes: 55`. Only these rules stop a
  case: FalkorDB's client runs without its 5 s socket timeout (`socket_timeout=None`), and Grafeo without its
  30 s query timeout (`query_timeout=None`, grafeo 0.6.1 and later); the other databases have no default timeout
  below the ceiling.

### Changed

- **Selections and silence limit**: `full` runs the medium scale only (push keeps small, under CodSpeed on every
  push); its warm-up, iteration and binding counts stay. The silence limit (`--case-timeout`) drops from 600 to
  120 s. A startup child keeps its own 120 s limit, no longer half the silence limit; the supervisor waits past
  it during the Startup row.
- **Fairness rules**: FAIRNESS.md is rewritten for the embedded comparison: one process per database,
  single-threaded runs (all-cores runs return with grafeo 0.6.4), in-memory storage where supported, each
  database's fastest documented bulk load, its vendor-recommended API, checked answers and the budget rule.
- **README and CONTRIBUTING**: rewritten for the new comparison and for adding a database adapter.
- **Dependencies**: the extras are `grafeo`, `ladybug`, `embedded` (both), `bench` (pytest-codspeed) and `dev`.

### Removed

- **The 0.2.x benchmark** and its unreleased 0.3 work (native and fallback labels, `vector_ann_*`, the Uni-DB
  adapter, the grafeo-server adapter) move to [legacy/](legacy/README.md), together with the server databases
  and the Docker setup. Results measured with it are not comparable with the new comparison.
- **The README's Latest Results section**, whose broken link was fixed twice: by
  [@0xdevalias](https://github.com/0xdevalias) in June and again by [@bitsnaps](https://github.com/bitsnaps) in
  October. 0.3.0 makes sure it never breaks again by deleting it. Thank you both.
