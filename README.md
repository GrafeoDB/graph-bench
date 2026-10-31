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
| Scans | label scan, property filter, aggregation, order and limit | property graph | planned |
| Hops | 1, 2 and 3 hop expansion, variable-length paths, shortest path | property graph | IC13 implemented, the rest planned |
| Pattern matching | triangles, cycles, multi-pattern joins | property graph | planned |
| Complex queries | LDBC SNB Interactive short reads IS1 to IS7 and complex reads | property graph | IS1 to IS7, IC2, IC6 implemented |
| Algorithms | Graphalytics BFS, WCC, PageRank, then CDLP, LCC, SSSP, Louvain, betweenness | databases with native algorithms and a property graph or SQL/PGQ model | BFS, WCC, PR implemented |
| Vector search | kNN, batch kNN, recall@k, filtered kNN | databases with vector indexes | planned |
| Hybrid | graph then vector, vector then graph | databases with both | planned |
| RDF/SPARQL | SNB IS1, IS4, IS6 in SPARQL, then the Berlin SPARQL Benchmark (BSBM) | Grafeo, Oxigraph | SNB implemented, BSBM planned |
| SQL/PGQ | SNB queries in SQL/PGQ, DuckPGQ's published queries | Grafeo, DuckDB + DuckPGQ | SNB IS1, IS4, IS6, IC13 implemented |
| Crash and restore | on disk: reopen, crash mid-transaction, rollback, snapshot isolation, torn WAL tail, backup and restore | all | planned |
| Deriva workload | a simulated multi-repo roundtrip using deriva's pipeline patterns | property graph | planned |
| Project suites | each project's own published queries, as written | the owning database and any that can run them | planned |

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
| ⏱ | too slow: beyond 10x the average of the 3 fastest other databases | `timeout` | failed, `too slow: ...` | Skipped if measured before, else absent |

Every benchmark first runs once, untimed: the preflight. The answer check and the time budget apply there, so a
wrong or too slow result never reaches CodSpeed as a speed. One exception: a ⏱ assigned only by the final budget
rule, applied after every database has run, comes after CodSpeed has measured the case. CodSpeed then shows that
case as measured, while results.json and the matrix show ⏱. A ⏱ from the in-run cutoff is never measured. The
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
- `full`: every category and scale, single-threaded. Feeds the [Results](#results) section.
- `release` (planned): `full` plus LDBC Graphalytics' wiki-Talk graph, checked against LDBC's published
  validation outputs, for Grafeo release runs.

`graph-bench codspeed --selection S --out DIR` is the CI entry point: one `pytest benches --codspeed` process
per database, under the CodSpeed runner. `graph-bench data key --selection S` prints the dataset cache key.
`GRAPH_BENCH_GRAFEO=<path>` installs that Grafeo at the start of `bench` and `codspeed`, like `--grafeo`.

### Grafeo's README summary

grafeo's README carries a summary block (between `<!-- graph-bench:start -->` and `<!-- graph-bench:end -->`)
with counts per category: how many databases took part, on how many benchmarks Grafeo was fastest, and its
median rank. It names no other database; this README names them all. At each Grafeo release the pre-release
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
  It runs the `push` selection on CodSpeed's
  Graviton macro runner (`codspeed-macro-arm64-graviton-ubuntu-22-04`, arm64) in wall-time mode on every push to `main`, daily and on demand. The daily run also runs `full`
  with `graph-bench bench` and opens a pull request on `bot/results` that updates the section below. The databases install one at a time
  (`.github/scripts/install-databases.sh`), so a package without a build for the runner does not stop the job: FalkorDB
  has no arm64 build for glibc 2.35 (falkordblite ships `manylinux_2_38` wheels only), so its cases show ✗ (not
  installed). Needs the
  CodSpeed app on graph-bench and codspeed-macro runners enabled for the org; authentication uses GitHub OIDC
  (no token secret).
- `.github/workflows/pins.yml`: weekly (and on demand), checks for a newer stable Neo4j Community release and a
  newer Temurin JRE of the Java major Neo4j requires, and opens a pull request on `bot/pins` that updates
  `graph_bench/databases/pins.py` (URLs and SHA-256 from the vendors' published checksums).
- `.github/workflows/release.yml`: the release post. Daily (and on demand) it reads the newest stable `grafeo`
  version on PyPI and compares it with the last one posted. When it is new, the job runs the `full` selection on the
  same Graviton runner as above (through the shared `.github/actions/prepare` action) and posts the trigger line,
  the overview per category, Grafeo's fastest counts and links to the run and to the results section on Discord
  (`graph-bench report --format discord`, `graph-bench notify`). The last posted version lives in the file
  `grafeo-version.txt` on the branch `bot/state`, written with the job's `GITHUB_TOKEN` only after a successful
  post. The webhook is the repository secret `DISCORD_BENCH_WEBHOOK`; without it the run skips the post, says so in
  the log and records nothing. A manual run has a `force` input that runs for the current version even if it was
  already posted.

## Results

<!-- results:start -->
No results yet: this section is written by the scheduled `full` run (`graph-bench report --format readme`).
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
