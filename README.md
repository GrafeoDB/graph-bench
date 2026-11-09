# graph-bench

[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://app.codspeed.io/GrafeoDB/graph-bench?utm_source=badge)

A head-to-head comparison of embedded graph databases. Every database runs in its own process, on the same
generated data, single-threaded, and every query answer is checked against a reference before any time
counts. A wrong answer is shown as wrong and never ranked.

graph-bench runs in its own CI on [CodSpeed](https://codspeed.io/) and locally with one command. Its results
feed the [Results](#results) section below and a summary block in [Grafeo](https://github.com/GrafeoDB/grafeo)'s
README at each Grafeo release. The rules every database is measured under are in [FAIRNESS.md](FAIRNESS.md).

## Databases

| Database | Package | Process | Status | Licence |
| --- | --- | --- | --- | --- |
| Grafeo | `grafeo` | in-process | implemented | Apache-2.0 |
| LadybugDB | `ladybug` | in-process | implemented | MIT |
| TuringDB | `turingdb` (embedded mode, Linux only) | in-process | implemented | BSL 1.1 (Community Edition) |
| FalkorDB | `falkordblite` (Linux only) | local process (a Redis server subprocess) | implemented | SSPL v1 (FalkorDB engine); BSD (falkordblite wrapper) |
| NeuG | `neug` | in-process | implemented | Apache-2.0 |
| ArcadeDB | `arcadedb-embedded` (bundles its JRE) | in-process (JVM), on disk | implemented | Apache-2.0 |
| Neo4j Community | `JPype1` (the Community tarball and a Temurin 21 JRE are downloaded) | in-process (JVM), on disk | implemented | GPLv3 (Neo4j Community and GDS Community) |
| DuckDB + DuckPGQ | `duckdb` (+ the `duckpgq` community extension) | in-process | implemented | MIT (DuckDB and duckpgq) |
| Oxigraph | `pyoxigraph` | in-process | implemented | MIT OR Apache-2.0 |
| HelixDB | `helix-db` + `helix-db-embedded` | in-process (embedded mode) | implemented | Apache-2.0 (engine and embedded runtime); Python SDK MIT |

graph-bench downloads each database at run time and redistributes none of them.

Neo4j is measured as Community only: the Enterprise terms restrict publishing benchmarks. Neo4j fetches its
Community tarball and a Temurin 21 JRE into the data cache (SHA-256 checked, about 690 MB with the extracted
copies); nothing is committed. Its algorithms use GDS Community from the same tarball. ArcadeDB's wheel bundles
its own JRE. Both JVM databases get the same heap cap (see FAIRNESS.md). HelixDB has no query text in its Python
SDK, so its queries are builders written with its Python DSL; the ones the DSL cannot express are marked not
available (see `docs/known-disagreements.md`).

## The matrix

The results are a capability matrix covering everything Grafeo does: one column per database, an overview with
one row per category, and one detail table per category with one row per benchmark. There is never one combined
score across categories. Each benchmark belongs to one data model (property graph, SQL/PGQ or RDF), which
decides the databases that take part; the others get ⊘. Property graph and SQL/PGQ queries read the same loaded
property graph; RDF queries read the dataset's RDF triples (`graph_bench/datasets/rdf.py`), a load of its own, so
the Load table has a separate row for it (⊘ for a database without RDF, as a load its models do not reach).

| Category | Benchmarks | Databases | Status |
| --- | --- | --- | --- |
| Load | bulk load per dataset (SNB, Graphalytics, SNB as RDF triples), each database's fastest documented bulk path | all | implemented |
| Load (query language) | bulk load through the query language (`LOAD CSV`, `COPY` from a query) | databases with a query-language loader | planned |
| Batch writes | batch insert, update, delete, upsert | property graph | planned |
| Scans | label scan, property filter, aggregation, order plus limit, property filter with average | property graph, plus SQL/PGQ and SPARQL twins | implemented |
| Hops | 1 and 2 hop expansion, variable length 1 to 3, shortest path | property graph, plus SQL/PGQ and SPARQL twins | implemented |
| Pattern matching | triangles through a person, a hub and the whole graph, 4-cycles, common friends, a multi-pattern join, whole-graph 2-hop counts, out-degree, trail semantics with self-loops (answer check only) | property graph, plus SQL/PGQ and SPARQL twins of the first four | implemented |
| Complex queries | LDBC SNB Interactive short reads IS1 to IS7 and complex reads | property graph | IS1 to IS7, IC2, IC6 implemented |
| Algorithms | Graphalytics BFS, WCC, PageRank, then CDLP, LCC, SSSP, Louvain, betweenness | databases with native algorithms and a property graph or SQL/PGQ model | BFS, WCC, PR implemented |
| Vector search | kNN, batch kNN, recall@k, filtered kNN | databases with vector indexes | planned |
| Hybrid | graph then vector, vector then graph | databases with both | planned |
| RDF/SPARQL | SNB IS1, IS4, IS6 in SPARQL, then the Berlin SPARQL Benchmark (BSBM) | Grafeo, Oxigraph | SNB implemented, BSBM planned |
| SQL/PGQ | SNB queries in SQL/PGQ, DuckPGQ's published queries | Grafeo, DuckDB + DuckPGQ | SNB IS1, IS4, IS6, IC13 implemented |
| Crash and restore | on disk: reopen, crash mid-transaction, rollback, snapshot isolation, torn WAL tail, backup and restore | all | planned |
| Deriva workload | a simulated multi-repo roundtrip using deriva's pipeline patterns | property graph | planned |
| Project suites | each project's own published queries, as written | the owning database and any that can run them | planned |

Memory and Disk are records beside the matrix, not timed rows: each load is repeated once in a fresh process whose
whole process tree (the interpreter, the started engine, an importer, a server that daemonizes) is read for its
resident memory, loaded and at its peak during the load, and the store's files are measured after the database has
closed. An in-memory store has no disk size. The details and their limits are in
[FAIRNESS.md](FAIRNESS.md#memory-and-disk); the files each on-disk database counts are in
[docs/known-disagreements.md](docs/known-disagreements.md).

Algorithms always use the database's native implementation, as a procedure or as a query in its own language. A
database without one gets ⊘; NetworkX is only the reference for the answer check.

### Markers

CodSpeed records a benchmark as measured or skipped, so each marker maps to a fixed pytest outcome in
`graph-bench codspeed`, with a reason prefix that shows in the CI log and in results.json.

| Marker | Meaning | results.json `status` / `answer` | pytest outcome | CodSpeed |
| --- | --- | --- | --- | --- |
| time, fastest **bold** | completed, answer checked | `ok` / `agree` | passed | measured |
| ⊘ | not available: the database lacks the capability or the data model | `not_supported` | skipped, `not available: ...` | absent |
| ✗ | failing: an error or a crash | `failed` | failed, `error: ...` | Skipped if measured before, else absent |
| ≠ | wrong answer: fails the answer check | `ok` / `disagree` | failed, `wrong answer: ...` | Skipped if measured before, else absent |
| ⏱ | too slow: beyond 10x the average of the 3 fastest other databases; `over 60 s`: a case still running 60 s after it started (the case ceiling); `load over 5 min`: a load still running 5 minutes after it started, per Load row (one dataset in one store), which takes the cases that read that Load row's data with it (the load ceiling) | `timeout` | failed, `too slow: ...` | Skipped if measured before, else absent |
| ⏱ `run deadline: 48 min` | not started: CI's run deadline had passed before the case, load or startup row could start (a run without `--deadline-s` has none) | `timeout` | skipped, `run deadline` | absent |

Every benchmark first runs once, untimed: the preflight. The answer check and the time budget apply there, so a
wrong or too slow result never reaches CodSpeed as a speed. One exception: a ⏱ assigned only by the final budget
rule, applied after every database has run, comes after CodSpeed has measured the case. CodSpeed then shows that
case as measured, while results.json and the matrix show ⏱. A ⏱ from the in-run cutoff or a ceiling is never
measured. The ceilings and the 5 s time box on timed runs are the same for every database
([FAIRNESS.md](FAIRNESS.md), rule 9). The
same holds for a load: its node and edge count check runs after CodSpeed timed it, so a load with ≠ shows as
measured in CodSpeed.

## Running it

```bash
git clone https://github.com/GrafeoDB/graph-bench
cd graph-bench
uv sync --extra embedded --extra bench --extra dev
```

`uv sync` installs Grafeo from PyPI. To measure another Grafeo, install it with `setup`, then use
`uv run --no-sync` so uv keeps it in place:

```bash
# A wheel, or a grafeo checkout built with maturin (set CARGO_TARGET_DIR to a drive with plenty of free space)
uv run --no-sync graph-bench setup --grafeo path/to/grafeo

# Generate the datasets and reference answers into the cache (GRAPH_BENCH_DATA_DIR, default ~/.cache/graph-bench)
uv run --no-sync graph-bench data prepare --selection push

# The plan: one benchmark id per line
uv run --no-sync graph-bench list --selection push

# Run every database, one worker process each, and write out/results.json
uv run --no-sync graph-bench bench --selection push --out out

# Only some databases, in this order
uv run --no-sync graph-bench bench --selection push --db grafeo,ladybug --out out

# The results as this README's section, or as the summary block for grafeo's README
uv run --no-sync graph-bench report --format readme --in out/results.json
uv run --no-sync graph-bench report --format summary --in out/results.json
```

Selections:

- `push`: small scale, the property graph core categories, single-threaded. Runs on every push to `main`.
- `full`: every category at medium scale, single-threaded. Feeds the [Results](#results) section. Small is
  covered by `push`, which CodSpeed runs on every push; `full` runs medium only so it fits the CodSpeed runner's
  hour. A row no database can measure on medium runs on small ([rule 9](FAIRNESS.md)): since the results below were
  measured, that is the SPARQL multi-pattern join, which the next full run reports under its own small-scale heading.
- `release` (planned): `full` plus LDBC Graphalytics' wiki-Talk graph, checked against LDBC's published
  validation outputs, for Grafeo release runs.

`graph-bench codspeed --selection S --out DIR` is the CI entry point: one `pytest benches --codspeed` process
per database, under the CodSpeed runner. `graph-bench data key --selection S` prints the dataset cache key.
`--yardstick FILE` (on `bench` and `codspeed`, repeatable) takes the results.json of earlier databases on the same
hardware: their correct results set the in-run budgets as earlier databases of the same run would. `graph-bench
merge --out DIR IN...` merges results.json files of runs over different databases (same selection, mode,
graph-bench commit and Grafeo) into one, as a single run over all of them would write it, and applies the final
budget rule once; with `--db`, a listed database without an input shows as "no results". `--local FILE` adds the
local snapshot's databases that are not in the inputs, tagged `local` (see [Results](#results)).
`graph-bench report --format overview --in FILE --write docs/results.md` writes the full overview page, and
`graph-bench local-check --local results/local/results.json` lists the snapshot's databases outside the roster
whose newest stable release (Neo4j: its pinned release) differs from the version the snapshot measured.
`GRAPH_BENCH_GRAFEO=<path>` installs that Grafeo at the start of `bench` and `codspeed`, like `--grafeo`.

### Grafeo's README summary

grafeo's README carries a summary block (between `<!-- graph-bench:start -->` and `<!-- graph-bench:end -->`)
with counts per category: how many databases took part, on how many benchmarks Grafeo was fastest, and its
median rank. It names no other database; this README names them all. Grafeo is ranked only against the databases
of its own run: with the local snapshot merged in, the header counts its databases, marked `†`, and a footer line
names its date and machine. The block links to `docs/results.md`. At each Grafeo release the pre-release
checklist runs, locally:

```bash
uv run --no-sync graph-bench setup --grafeo path/to/grafeo.whl --grafeo-sha <commit>
uv run --no-sync graph-bench bench --selection full --out out
uv run --no-sync graph-bench report --format summary --in out/results.json
```

and commits the block on grafeo's release branch. No automation writes to grafeo. `--grafeo-sha` records the
grafeo commit the wheel was built from, since a wheel does not name it and local builds of a release branch share
one version string.

### CI

- `.github/workflows/ci.yml`: lint and the test suite on Python 3.12 and 3.13, including the adapter tests
  against Grafeo and LadybugDB from PyPI.
- `.github/workflows/bench.yml`: graph-bench's own CodSpeed project, separate from grafeo's CodSpeed benchmarks,
  which it does not replace. It measures the newest stable release of every database from PyPI, grafeo
  included (grafeo's own CodSpeed suite covers `main`): after the locked `uv sync` it upgrades the database
  packages (`graph-bench packages`) without touching `uv.lock`, and the job log lists the installed versions.
  It runs the `push` selection on CodSpeed's Graviton macro runner (`codspeed-macro-arm64-graviton-ubuntu-22-04`,
  arm64) in wall-time mode on every push to `main`, daily and on demand. The daily run also runs `full` with
  `graph-bench bench` and opens a pull request on `bot/results` that updates the section below. The databases
  install one at a time (`.github/scripts/install-databases.sh`), so a package without a build for the runner
  does not stop the job. Needs the CodSpeed app on graph-bench and codspeed-macro runners enabled for the org;
  authentication uses GitHub OIDC (no token secret).
- **How the results are produced.** CI runs a fixed roster, `catalog.CI_DATABASES`: NeuG, LadybugDB, TuringDB,
  Oxigraph, DuckDB + DuckPGQ and Grafeo, in that order. The macro runner dies about an hour after boot, so each
  selection is a matrix with one job per database, each on a fresh box, one at a time: Grafeo runs last, so its
  in-run budgets come from the others' results on the same hardware (each job passes the results of the jobs
  before it in the roster as `--yardstick`, never a later one's, even on a re-run). The grafeo version is resolved
  once per run and every job installs exactly it. Every job stops starting cases at the run deadline
  (`--deadline-s`: what is left of 48 minutes counted from the job's start, so setup counts too, under the job's
  55-minute timeout), so it always leaves its results.json; the cases it did not reach are ⏱ `run deadline: 48 min`
  (`--deadline-total-s` names the whole deadline, not what was left). A final job on `ubuntu-latest` merges the jobs' results (`graph-bench merge`), which
  applies the 10x budget rule once over all databases, so the markers are those of one run over the whole roster;
  a job that left no results shows its database as ✗ "no results". The README job then mixes in the committed
  local snapshot of all ten (`graph-bench merge --local`; its non-roster databases only, marked `†`), writes the
  section and `docs/results.md`, and raises a notice listing the snapshot's databases whose newest stable release
  differs from the version it measured (`graph-bench local-check`). A cancelled run merges nothing. A release run
  whose merged results hold no Grafeo posts and records nothing, so the next daily run tries again. CodSpeed merges
  the push jobs of one run itself. One concurrency group,
  `macro-runner`, on both bench.yml and release.yml keeps their macro-runner jobs from ever running at once; a
  newer queued run replaces an older pending one.
- `.github/workflows/pins.yml`: weekly (and on demand), checks for a newer stable Neo4j Community release and a
  newer Temurin JRE of the Java major Neo4j requires, and opens a pull request on `bot/pins` that updates
  `graph_bench/databases/pins.py` (URLs and SHA-256 from the vendors' published checksums).
- `.github/workflows/release.yml`: the release post. Daily (and on demand) it reads the newest stable `grafeo`
  version on PyPI and compares it with the last one posted. When it is new, it runs the `full` selection on the
  same Graviton runner as above, one job per roster database (through the shared `.github/actions/prepare` action),
  merges their results on `ubuntu-latest` and posts the trigger line,
  the overview per category, Grafeo's fastest counts and links to the run and to the results section on Discord
  (`graph-bench report --format discord`, `graph-bench notify`). The last posted version lives in the file
  `grafeo-version.txt` on the branch `bot/state`, written with the job's `GITHUB_TOKEN` only after a successful
  post. The webhook is the repository secret `DISCORD_BENCH_WEBHOOK`; without it the run skips the post, says so in
  the log and records nothing. A manual run has a `force` input that runs for the current version even if it was
  already posted.

## Results

The table mixes two sources, never compared with each other. The CI columns come first: Grafeo and the fixed
roster, measured by the scheduled `full` run on the Graviton runner. The other databases follow, each marked `†`:
they come from the local snapshot of all ten (`results/local/results.json`, refreshed by hand, see
CONTRIBUTING.md), and the footnote names its date and machine. Budgets and bold (the fastest) apply within each
source only. [docs/results.md](docs/results.md) lists every database's version, source, machine and run date, with
the complete per-case tables.

<!-- results:start -->
### Overview

Single-threaded.

✓ k/n: correct answers within the time budget, out of the category's n benchmarks; then the count of each marker.

† From the local snapshot of 2026-11-02 on AMD Ryzen 7 7800X3D 8-Core Processor, 16 cores, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39, not from CI: these columns are budgeted and ranked only among themselves, never against the CI columns, and bold marks the fastest of each group.

| Category | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| Startup | ✓ 1/1 | ✓ 1/1 | ✓ 1/1 | ✓ 0/1 ≠1 | ✓ 1/1 | ✓ 1/1 | ✓ 1/1 | ✓ 1/1 | ✓ 1/1 | ✓ 1/1 |
| Load, medium | ✓ 3/3 | ✓ 2/3 ⊘1 | ✓ 2/3 ⊘1 | ✓ 2/3 ⊘1 | ✓ 1/3 ⊘2 | ✓ 2/3 ⊘1 | ✓ 2/3 ⊘1 | ✓ 2/3 ⊘1 | ✓ 2/3 ⊘1 | ✓ 2/3 ⊘1 |
| Load (query language), medium | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Scans, medium | ✓ 15/15 | ✓ 5/15 ⊘10 | ✓ 5/15 ⊘10 | ✓ 5/15 ⊘10 | ✓ 5/15 ⊘10 | ✓ 5/15 ⊘10 | ✓ 5/15 ⊘10 | ✓ 5/15 ⊘10 | ✓ 0/15 ⊘11 ⏱4 | ✓ 3/15 ⊘10 ⏱2 |
| Hops, medium | ✓ 9/9 | ✓ 1/9 ⊘6 ⏱2 | ✓ 2/9 ⊘5 ≠1 ⏱1 | ✓ 3/9 ⊘5 ⏱1 | ✓ 2/9 ⊘7 | ✓ 2/9 ⊘5 ✗1 ⏱1 | ✓ 4/9 ⊘5 | ✓ 2/9 ⊘5 ✗1 ⏱1 | ✓ 0/9 ⊘5 ⏱4 | ✓ 4/9 ⊘5 |
| Pattern matching, small | ✓ 1/1 | ⊘ | ⊘ | ⊘ | ✓ 1/1 | ⊘ |  |  |  |  |
| Pattern matching, medium | ✓ 8/18 ⏱9 | ✓ 3/18 ⊘13 ✗1 | ✓ 9/18 ⊘7 ≠1 | ✓ 8/18 ⊘7 ≠1 ⏱1 | ✓ 1/18 ⊘14 ⏱2 | ✓ 4/18 ⊘7 ≠1 ⏱5 | ✓ 4/18 ⊘8 ⏱6 | ✓ 3/18 ⊘8 ≠1 ⏱6 | ✓ 1/18 ⊘15 ≠1 ⏱1 | ✓ 6/18 ⊘8 ⏱4 |
| Complex queries, medium | ✓ 9/9 | ⊘ | ✓ 9/9 | ✓ 9/9 | ⊘ | ✓ 8/9 ⏱1 | ✓ 7/9 ⏱2 | ✓ 5/9 ⏱4 | ✓ 2/9 ⊘3 ⏱4 | ✓ 5/9 ⏱4 |
| Algorithms, medium | ✓ 3/3 | ✓ 1/3 ⏱2 | ✓ 1/3 ✗2 | ✓ 2/3 ⏱1 | ⊘ | ⊘ | ✓ 1/3 ⏱2 | ✓ 2/3 ⏱1 | ⊘ | ✓ 1/3 ⏱2 |
| RDF/SPARQL, medium | ✓ 3/3 | ⊘ | ⊘ | ⊘ | ✓ 3/3 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| SQL/PGQ, medium | ✓ 3/4 ✗1 | ✓ 0/4 ✗2 ⏱2 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |

### Startup (empty database, ms)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| Empty database | **12.4** | 213 | 128 | ≠ | 14.2 | 1456 | 689 | **146** | 153 | 6055 |

### Load (medium scale, s)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| LDBC SNB | 1.89 | **1.25** | 2.02 | 5.59 | ⊘ | 5.63 | 3.82 | **3.12** | 70.0 | 10.4 |
| Graphalytics | 0.867 | **0.321** | 1.01 | 2.07 | ⊘ | 3.43 | **1.05** | 12.1 | 53.8 | 10.2 |
| LDBC SNB (RDF) | 2.96 | ⊘ | ⊘ | ⊘ | **2.37** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |

### Load (query language) (medium scale, s)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| LDBC SNB | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Graphalytics | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |

### Scans (medium scale, ms)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| Label scan | **50.3** | ⊘ | 87.3 | 57.1 | ⊘ | 172 | 12.2 | **9.12** | ⏱ | ⏱ |
| Property filter | 180 | ⊘ | 194 | **113** | ⊘ | 420 | 94.5 | **57.1** | ⏱ | ⏱ |
| Aggregation | 12.8 | ⊘ | **5.02** | 11.7 | ⊘ | 8.50 | 5.59 | **2.69** | ⊘ | 11.1 |
| Order plus limit | 83.8 | ⊘ | 40.2 | **3.28** | ⊘ | 62.2 | 55.2 | **19.8** | ⏱ | 42.7 |
| Property filter with average | 76.6 | ⊘ | **8.01** | 11.6 | ⊘ | 65.4 | 50.9 | **30.6** | ⏱ | 47.1 |
| Label scan (SQL/PGQ) | 50.7 | **15.9** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Property filter (SQL/PGQ) | 181 | **75.2** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Aggregation (SQL/PGQ) | 12.5 | **4.61** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Order plus limit (SQL/PGQ) | 194 | **38.5** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Property filter with average (SQL/PGQ) | 74.2 | **36.0** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Label scan (SPARQL) | **189** | ⊘ | ⊘ | ⊘ | 389 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Property filter (SPARQL) | **2701** | ⊘ | ⊘ | ⊘ | 3364 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Aggregation (SPARQL) | **121** | ⊘ | ⊘ | ⊘ | 241 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Order plus limit (SPARQL) | **1505** | ⊘ | ⊘ | ⊘ | 1759 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Property filter with average (SPARQL) | **759** | ⊘ | ⊘ | ⊘ | 1382 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |

### Hops (medium scale, ms)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-hop | 3.68 | ⊘ | 8.42 | **2.32** | ⊘ | ⏱ | **2.04** | 7.02 | ⏱ | 4.34 |
| 2-hop | 250 | ⊘ | 186 | **131** | ⊘ | 160 | 102 | **43.2** | ⏱ | 73.6 |
| Variable length 1 to 3 | 945 | ⊘ | ⏱ | ⏱ | ⊘ | **530** | 470 | ⏱ | ⏱ | **342** |
| 1-hop (SQL/PGQ) | **3.52** | ⏱ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| 2-hop (SQL/PGQ) | **198** | 1011 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Variable length 1 to 3 (SQL/PGQ) | **827** | ⏱ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| 1-hop (SPARQL) | **21.2** | ⊘ | ⊘ | ⊘ | 42.6 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| 2-hop (SPARQL) | **1413** | ⊘ | ⊘ | ⊘ | 3375 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| IC13 shortest path | **19.7** | ⊘ | ≠ | 117 | ⊘ | ✗ | **3.32** | ✗ | ⏱ | 6.48 |

### Pattern matching (small scale, ms)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| Multi-pattern join (SPARQL) | 8058 | ⊘ | ⊘ | ⊘ | **2469** | ⊘ |  |  |  |  |

### Pattern matching (medium scale, ms)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| Triangles through a person | 3319 | ⊘ | 38.0 | **10.9** | ⊘ | 1247 | ⏱ | 59.3 | ⊘ | **28.3** |
| 4-cycles through a person | ⏱ | ⊘ | 1455 | **812** | ⊘ | ⏱ | ⏱ | ⏱ | ⊘ | **129** |
| Common friends | 121 | ⊘ | **21.9** | 63.9 | ⊘ | 96.8 | ⏱ | 28.1 | 39.2 | **4.05** |
| Multi-pattern join | ⏱ | ⊘ | 92.6 | **6.31** | ⊘ | 20.6 | **49.7** | ⏱ | ⊘ | 60.4 |
| Triangles through a hub person | ⏱ | ⊘ | 2056 | **321** | ⊘ | ⏱ | **1976** | ⏱ | ⊘ | ⏱ |
| Triangles, whole graph | ⏱ | ⊘ | **3177** | 3587 | ⊘ | ⏱ | ⏱ | ⏱ | ⊘ | ⏱ |
| 2-hop paths, whole graph | ⏱ | ⊘ | **20.1** | 156 | ⊘ | ⏱ | ⏱ | ⏱ | ⏱ | ⏱ |
| 2-hop paths per source, top 10 | ⏱ | ⊘ | **12479** | ⏱ | ⊘ | ⏱ | ⏱ | ⏱ | ⊘ | ⏱ |
| Out-degree, top 10 | 43.4 | ⊘ | **18.4** | 35.7 | ⊘ | 54.9 | **32.5** | 52.3 | ⊘ | 54.1 |
| Trail semantics with self-loops | ✓ | ⊘ | ≠ | ≠ | ⊘ | ≠ | ✓ | ≠ | ≠ | ✓ |
| Triangles through a person (SQL/PGQ) | 1855 | **1151** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| 4-cycles through a person (SQL/PGQ) | ⏱ | **2721** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Common friends (SQL/PGQ) | **91.1** | 531 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Multi-pattern join (SQL/PGQ) | ⏱ | ✗ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Triangles through a person (SPARQL) | 12295 | ⊘ | ⊘ | ⊘ | **2284** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| 4-cycles through a person (SPARQL) | ⏱ | ⊘ | ⊘ | ⊘ | ⏱ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Common friends (SPARQL) | **31.5** | ⊘ | ⊘ | ⊘ | ⏱ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| Multi-pattern join (SPARQL) |  |  |  |  |  |  | ⊘ | ⊘ | ⊘ | ⊘ |

### Complex queries (medium scale, ms)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| IS1 person profile | 2.43 | ⊘ | 4.28 | **2.20** | ⊘ | ⏱ | **1.14** | ⏱ | ⏱ | ⏱ |
| IS2 recent messages of a person | 4.88 | ⊘ | 507 | **4.51** | ⊘ | 156 | **2.84** | 7.19 | ⏱ | 8.72 |
| IS3 friends of a person | **10.5** | ⊘ | 18.0 | 12.4 | ⊘ | 121 | **8.61** | 9.41 | ⊘ | ⏱ |
| IS4 message content | 2.01 | ⊘ | 68.8 | **1.62** | ⊘ | 74.3 | ⏱ | ⏱ | ⏱ | **4.17** |
| IS5 message creator | 1.94 | ⊘ | 37.4 | **1.57** | ⊘ | 75.5 | ⏱ | ⏱ | **32.3** | ⏱ |
| IS6 forum of a message | 2.92 | ⊘ | 215 | **2.27** | ⊘ | 112 | 109 | ⏱ | **46.2** | ⏱ |
| IS7 replies of a message | 5.87 | ⊘ | 41.7 | **3.84** | ⊘ | 147 | **2.37** | 9.30 | ⊘ | 5.50 |
| IC2 recent messages by friends | 15.5 | ⊘ | 96.3 | **7.74** | ⊘ | 142 | 21.1 | **10.2** | ⏱ | 29.0 |
| IC6 tag co-occurrence | 505 | ⊘ | 636 | **188** | ⊘ | 431 | 431 | 548 | ⊘ | **318** |

### Algorithms (medium scale, ms)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| BFS | **25.7** | ⏱ | 159 | 51.8 | ⊘ | ⊘ | ⏱ | ⏱ | ⊘ | ⏱ |
| WCC | 84.7 | ⏱ | ✗ | **27.2** | ⊘ | ⊘ | ⏱ | **14.9** | ⊘ | 107 |
| PageRank | **82.2** | 132 | ✗ | ⏱ | ⊘ | ⊘ | 121 | **15.6** | ⊘ | ⏱ |

### RDF/SPARQL (medium scale, ms)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| IS1 person profile | **12.9** | ⊘ | ⊘ | ⊘ | 14.0 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| IS4 message content | **6.86** | ⊘ | ⊘ | ⊘ | 7.61 | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| IS6 forum of a message | 11.2 | ⊘ | ⊘ | ⊘ | **10.7** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |

### SQL/PGQ (medium scale, ms)

| Benchmark | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| IS1 person profile | **2.36** | ⏱ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| IS4 message content | **1.93** | ✗ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| IS6 forum of a message | ✗ | ✗ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| IC13 shortest path | **21.1** | ⏱ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |

### Memory (medium scale, MiB)

| Dataset | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| LDBC SNB, loaded | 93.8 | 61.7 | 124 | **29.0** | ⊘ | 185 | 851 | 434 | 684 | **83.6** |
| LDBC SNB, peak during load | 153 | 145 | 182 | **136** | ⊘ | 358 | 1007 | **488** | 740 | 5021 |
| Graphalytics, loaded | 40.7 | 38.9 | 98.3 | **18.9** | ⊘ | 85.2 | 308 | 353 | 575 | **112** |
| Graphalytics, peak during load | **107** | 122 | 157 | 129 | ⊘ | 258 | 543 | **407** | 630 | 5022 |
| LDBC SNB (RDF), loaded | 462 | ⊘ | ⊘ | ⊘ | **277** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |
| LDBC SNB (RDF), peak during load | 587 | ⊘ | ⊘ | ⊘ | **365** | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |

"Loaded" is the memory the load added (after minus before); "peak" is the process tree's absolute peak during the load, interpreter and started engine included.

### Disk (medium scale, MiB)

| Dataset | Grafeo | DuckDB + DuckPGQ | LadybugDB | NeuG | Oxigraph | TuringDB | ArcadeDB† | FalkorDB† | HelixDB† | Neo4j† |
|---|---|---|---|---|---|---|---|---|---|---|
| LDBC SNB | in memory | in memory | in memory | in memory | ⊘ | **128** | **27.3** | in memory | in memory | 545 |
| Graphalytics | in memory | in memory | in memory | in memory | ⊘ | **60.0** | **17.9** | in memory | in memory | 529 |
| LDBC SNB (RDF) | in memory | ⊘ | ⊘ | ⊘ | in memory | ⊘ | ⊘ | ⊘ | ⊘ | ⊘ |

⊘ not available · ✗ failing · ≠ wrong answer · ⏱ beyond 10x the top-3 average, over 60 s, load over 5 min or not started by the run deadline: 48 min

✓: an answer-check-only case agrees; it has no time budget and is not ranked.

graph-bench 20f6913 · Grafeo 0.6.1 (pypi) · aarch64, 16 cores, Linux-6.5.0-1024-aws-aarch64-with-glibc2.35 · 2026-11-04 · selection full · bench mode

Local snapshot†: graph-bench ebfbf59 · Grafeo 0.6.1 (pypi) · AMD Ryzen 7 7800X3D 8-Core Processor, 16 cores, Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.39 · 2026-11-02 · selection full · bench mode

| Database | Version | Process | Storage | Threads |
|---|---|---|---|---|
| Grafeo | 0.6.1 | in-process | in-memory | single-threaded executor |
| DuckDB + DuckPGQ | 2.0.0 | in-process | in-memory | SET threads=1 |
| LadybugDB | 0.22.1 | in-process | in-memory | max_num_threads=1 |
| NeuG | 0.3.0 | in-process | in-memory | max_thread_num=1; embedded queries are single-threaded |
| Oxigraph | 0.5.13 | in-process | in-memory | bulk_load uses several threads internally, not configurable; queries run on the calling thread |
| TuringDB | 3.0.2 | in-process | on-disk | TuringDB exposes no thread setting |
| ArcadeDB† | 26.10.2.post1 (Java 25.0.4.1) | in-process | on-disk | configured to 1 thread: arcadedb.queryParallelismPoolThreads=1, arcadedb.parallelScanProducerPoolThreads=1, arcadedb.queryParallelScan=false, arcadedb.asyncWorkerThreads=1, java.util.concurrent.ForkJoinPool.common.parallelism=1, GraphBatch parallel_flush=False; JVM: -XX:ParallelGCThreads=1, -XX:ConcGCThreads=1, -XX:CICompilerCount=2 (default collector and tiered compilation kept; 2 is HotSpot's minimum compiler count with tiered compilation, one C1 and one C2 thread); not capped: the JVM's other service threads (signal dispatcher, reference handler, finalizer) and ArcadeDB's housekeeping threads |
| FalkorDB† | 0.11.0 | local process | in-memory | THREAD_COUNT and OMP_THREAD_COUNT are module load arguments that falkordblite does not expose |
| HelixDB† | 0.3.4 | in-process | in-memory | HelixDB exposes no thread setting |
| Neo4j† | 2026.10.1 (Java Temurin 21.0.13) | in-process | on-disk | configured to 1 thread: Cypher runs each query on the calling thread (Community has no parallel runtime); GDS concurrency=1 and readConcurrency=1; neo4j-admin import --threads=1; JVM (embedded and importer): -XX:ParallelGCThreads=1, -XX:ConcGCThreads=1, -XX:CICompilerCount=2; not capped: Neo4j's background threads (page cache flushing, checkpointing, index population) and the JVM's other service threads, which have no documented setting |

Every database's version, source, machine and run date: [docs/results.md](docs/results.md)
<!-- results:end -->

## Documentation

- [FAIRNESS.md](FAIRNESS.md): the rules every database is measured under.
- [CONTRIBUTING.md](CONTRIBUTING.md): adding a database, running the tests and lint.
- [docs/known-disagreements.md](docs/known-disagreements.md): wrong answers, slow cases and engine findings,
  recorded as they are instead of worked around.
- [CHANGELOG.md](CHANGELOG.md).

The pre-0.3.0 benchmark, including the server databases, is kept in [legacy/](legacy/README.md).

## LDBC disclaimer

Several categories use workloads **inspired by** [LDBC](https://ldbcouncil.org/) specifications: the
[Social Network Benchmark (SNB)](https://ldbcouncil.org/benchmarks/snb/) Interactive queries and
[Graphalytics](https://ldbcouncil.org/benchmarks/graphalytics/). These are **not official LDBC Benchmark
results** and have not been audited or endorsed by the LDBC Council. Differences from official LDBC benchmarks:

- **No LDBC audit**: official LDBC results require formal auditing under the
  [LDBC Byelaws](https://ldbcouncil.org/benchmarks/fair-use-policies/).
- **Synthetic datasets**: graph-bench generates its own data following the SNB schema, not with the official
  LDBC SNB data generator (Datagen). Graphalytics runs on the SNB `KNOWS` graph.
- **Reduced scale**: far below the scale factors used in official LDBC publications.
- **Subset of queries**: only part of the SNB Interactive and Graphalytics workloads is implemented.
- **No cost-of-ownership reporting**: official LDBC results include pricing and hardware disclosure.

LDBC specifications are used under the [Creative Commons Attribution 4.0 (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. LDBC and the LDBC logo are trademarks of the Linked Data Benchmark Council. See the
[LDBC fair use policy](https://ldbcouncil.org/benchmarks/fair-use-policies/).

## License

Apache 2.0
