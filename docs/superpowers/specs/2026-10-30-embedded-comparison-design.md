# Embedded graph database comparison: design

Status: agreed 2026-10-30 between the user, grafeo-31 (grafeo repo side) and graph-bench-83 (this repo).

## Goal

Rebuild graph-bench as a head-to-head comparison of ten embedded graph databases. It runs in graph-bench's own CI on CodSpeed and locally with one command. It feeds full results to graph-bench's README and a summary block to grafeo's README at each grafeo release.

The new code fully replaces the current benchmark. The current code moves to `legacy/` untouched and is deleted once everything is replaced. The server comparison (Neo4j, Memgraph, FalkorDB, TuGraph, grafeo-server) is out of scope. It will later follow the same pattern, triggered from the grafeo-server repo.

## Databases

| Database | Package | Process | Notes |
|---|---|---|---|
| Grafeo | `grafeo` | in-process | query executor is single-threaded until parallel queries land (0.6.4); only rayon parts (vector batch, clustering, folds) are parallel, set with `RAYON_NUM_THREADS` |
| LadybugDB | `ladybug` | in-process | `max_num_threads`, `buffer_pool_size`; ALGO extension for algorithms |
| TuringDB | `turingdb` | in-process | new adapter for `TuringDB(type="embedded", data_dir=...)`; HTTP adapter is not comparable |
| FalkorDB | `falkordblite` | local process | runs a Redis server subprocess, reported as "local process" |
| NeuG | `neug` 0.3.0 | in-process | |
| ArcadeDB | `arcadedb-embedded` | in-process (JVM) | bundles its own JRE |
| Neo4j Community | JPype + Maven jars | in-process (JVM) | Community only (Enterprise terms restrict publishing benchmarks); jars and a Temurin 21 JRE fetched into the data cache, never committed |
| DuckDB + DuckPGQ | `duckdb` | in-process | SQL/PGQ only |
| Oxigraph | `pyoxigraph` | in-process | RDF only |
| HelixDB | `helix-db` + `helix-db-embedded` (native wheel) | in-process (embedded mode) | graph + vector database with its own query language (HelixQL); added by the user 2026-10-30. Uses `Client.embedded(...)` with in-memory storage (docs: https://docs.helix-db.com/database/helix-db/start-here/local-development/embedded-database.md). If the embedded wheel is unusable on the benchmark platforms, HelixDB is deferred (server-only databases are not part of the release run) |

New dependencies in graph-bench need no approval (user, 2026-10-30).

## The matrix

The output is a capability matrix covering everything Grafeo does. Columns are the databases. The README shows:

- an **overview**: one row per category, each cell showing how many of that category's benchmarks the database completed correctly, plus counts per marker;
- **one detail table per category**: one row per benchmark, each cell the median time (fastest in bold) or a marker.

There is never one combined score across categories.

Each benchmark belongs to one category and one data model (property graph, SQL/PGQ, RDF). The data model decides which databases can take part; everything else gets the "not available" marker.

| Category | Benchmarks (first plan in brackets) | Databases |
|---|---|---|
| Load | bulk load per dataset [SNB, Graphalytics] | all |
| Load (query language) | bulk load through the database's query language where it has one (Grafeo `LOAD CSV`, Neo4j `LOAD CSV`, Kùzu-family `COPY` from a query), tracked separately so engine-side load fixes show up; the Load row above uses each database's fastest documented bulk path | DBs with a query-language loader |
| Batch writes | batch insert, update, delete, upsert | property graph DBs |
| Scans | label scan, property filter, aggregation, order + limit | property graph DBs |
| Hops | 1, 2, 3 hop expansion, variable-length paths, shortest path [IC13] | property graph DBs |
| Pattern matching | triangles, cycles, multi-pattern joins | property graph DBs |
| Complex queries | LDBC SNB Interactive short reads IS1-7 and complex reads [IS1-7, IC2, IC6] | property graph DBs |
| Algorithms | Graphalytics BFS, WCC, PR [these three], then CDLP, LCC, SSSP, Louvain, betweenness | DBs with native algorithms |
| Vector search | kNN, batch kNN, recall@k, filtered kNN | DBs with vector indexes |
| Hybrid | graph then vector, vector then graph | DBs with both |
| RDF/SPARQL | Berlin SPARQL Benchmark (BSBM) | Grafeo, Oxigraph |
| SQL/PGQ | SNB queries in SQL/PGQ, DuckPGQ's published queries | Grafeo, DuckDB+DuckPGQ |
| Crash and restore | in on-disk mode, regardless of the in-memory default: reopen after a clean close (data intact, time to reopen); a crash mid-transaction via a hard kill of the worker, then reopen (committed data present, uncommitted absent: atomicity and durability); a failed batch rolls back fully; snapshot isolation of a reader during a write; recovery after a torn WAL tail; on-disk load and file size; backup and restore, timed (the database's own backup or export, then restore into a fresh directory, then an answer check that the restored graph equals the original; ⊘ if it has no backup or export). Built from the legacy `ldbc_acid.py` checks. A database with no on-disk mode gets ⊘; one that loses or corrupts data gets ≠; one that won't reopen or crashes gets ✗ | all |
| Deriva workload | simulated multi-repo roundtrip using deriva's pipeline patterns: row-by-row upserts, sweeps, repo delete, filter-heavy candidate reads, 2-hop, prep algorithms with write-back, Model writes with dedupe, export. Profile: `docs/deriva-workload-profile.md` | property graph DBs |
| Project suites | each project's own published queries, as written, with source and version: Ladybug (from Kùzu), TuringDB, FalkorDB, DuckPGQ, NeuG; Oxigraph's is BSBM above | the owning DB plus any DB that can run them |

Algorithms use each database's native implementation, as a procedure or as a query in its own language. Graph-bench never falls back to NetworkX inside the comparison: a database without one is "not available". NetworkX is only the reference for the answer check.

### Markers and their CodSpeed status

CodSpeed records a benchmark as either measured or skipped. When a benchmark that ran before is missing from a run, CodSpeed shows it as Skipped ⏩ and keeps its baseline. So the four markers map to different pytest outcomes with a fixed reason prefix (visible in the CI log and results.json), and each lands as "Skipped" or "absent" in CodSpeed:

| Marker | Meaning | results.json `status` / `answer` | pytest outcome (codspeed mode) | CodSpeed |
|---|---|---|---|---|
| time, fastest **bold** | completed, answer checked | `ok` / `agree` | passed | measured: New, Improved, Regressed or Untouched |
| ⊘ | not available: the database lacks the capability or the data model | `not_supported` | skipped, reason `not available: ...` | absent (never measured) |
| ✗ | failing: error or crash | `failed` | failed, reason `error: ...` | Skipped ⏩ if measured before, else absent |
| ≠ | wrong answer: fails the answer check | `ok` / `disagree` | failed, reason `wrong answer: ...` | Skipped ⏩ if measured before, else absent |
| ⏱ | too slow: did not finish within 10x the average of the top 3 | `timeout` | failed, reason `too slow: ...` | Skipped ⏩ if measured before, else absent |

One exception: a ⏱ that only the final rule assigns (applied after every database has run, see fairness rule 8) comes after CodSpeed has already measured the case. CodSpeed then shows that case as measured, while results.json and the matrix show ⏱. A ⏱ from the in-run cutoff is never measured.

Each benchmark first runs once, untimed: the **preflight**. The answer check and the time budget are applied there, and only a benchmark that passes the preflight is timed, so a wrong or too-slow result never reaches CodSpeed as a speed. The preflight also counts as the first warm-up iteration.

## Fairness rules

These go into FAIRNESS.md, which is trimmed to embedded use.

1. Each database runs in its own process. A Python process can host only one JVM, and separate processes keep allocators and caches from interfering.
2. JVM databases get the same warm-up and memory cap as the others, and JVM startup is not timed.
3. Every database runs single-threaded. A database whose thread count can't be set is marked as such. Each database's `threads_note` in results.json says how the setting was applied. Grafeo gets `RAYON_NUM_THREADS=1` in its worker, note "single-threaded executor".
   - **Planned with grafeo 0.6.4 (parallel queries):** add the all-cores runs back to `full`, reported as a separate set of tables, and switch Grafeo's adapter from `RAYON_NUM_THREADS` to the real thread setting 0.6.4 brings. The selection and benchmark ids already carry a threads value (`-allcores` suffix) so this is a data change, not a redesign.
4. Storage is in-memory where the database supports it; a disk-only database runs on disk and is labelled so (user decision, 2026-10-30).
   Loading is measured separately from queries. Each database loads the same CSV files with its own documented bulk-load path.
5. Answers are checked. Each query's answer is compared with a Python reference computed from the generated data (NetworkX for algorithms). A database with a different answer is marked `disagree` and left out of the ranking.
6. Queries return scalar values only (no node or edge objects) and order fully, with tie-breakers, whenever they use `LIMIT`.
7. PageRank is compared at convergence (damping 0.85), with the LDBC relative epsilon of 0.0001.
8. The time budget per benchmark is 10x the average median of the 3 fastest *other* databases with correct answers (the average over those available if fewer than 3; no budget when no other database has a correct result). This final rule, on medians, decides the ⏱ marker.
   - **During a run, a runaway guard:** databases run one after another. Once at least 3 earlier databases have correct times, a later database's preflight, its first (cold) run, is cut off at 30x the average of their 3 fastest preflights: the supervisor kills the worker as soon as the preflight runs past that budget and records ⏱ with "too slow: cut off at <b> s (30x the top-3 average of earlier databases)". The cutoff only stops runaway cases early; because it measures cold first runs, it is set well above the final rule. Before 3 earlier results exist, only a hard cap applies (`--case-timeout`, default 600 s, measured as silence: the worker reports progress after every run).
   - **After the run:** the 10x rule is applied over all results, each entry against the other databases. Any benchmark beyond its budget becomes ⏱, so the run order never decides the marker.
   - Loads are timed but not answer-checked and are exempt from this rule; a node and edge count check is planned with the next adapters.

## Datasets

- Generated from a fixed seed or downloaded, never committed.
- Cache: `GRAPH_BENCH_DATA_DIR`, default `~/.cache/graph-bench`.
- Each dataset is a directory of CSV files: one per node label and one per (edge type, source label, target label). A `schema.json` gives the typed columns, and `references/` holds the per-case reference answers.
- Dates are stored as integer epoch milliseconds, so date types never decide an answer.
- `graph-bench data key --selection S` prints a hash of the generator source, the parameters, the seeds and the pinned JRE/jar versions, for `actions/cache`.
- Graphalytics uses the SNB `KNOWS` graph stored in both directions, so directed algorithm implementations give the undirected answer.
- **wiki-Talk (release runs):** LDBC Graphalytics' wiki-Talk (about 2.4M vertices, 5M directed edges, unweighted) is downloaded from LDBC's published archive and cached under its own key. Its references are LDBC's published validation outputs for BFS, WCC, PageRank, CDLP and LCC (not NetworkX, which is too slow at this size); SSSP is skipped (no weights). It is directed, so the algorithms follow the LDBC Graphalytics directed definitions. It belongs to a separate `release` selection (= `full` + wiki-Talk), so `full` stays quick for local runs. Release runs get a generous hard cap per case; the 10x rule is the only ⏱ criterion (user decision, 2026-10-30).

## Interface (agreed with grafeo-31)

| Command | Purpose |
|---|---|
| `graph-bench setup --grafeo <wheel-or-checkout>` | installs a grafeo wheel, or builds one from a checkout with maturin; writes `.graph-bench/grafeo.json` (version, wheel sha256, git sha). `GRAPH_BENCH_GRAFEO=<path>` does the same at the start of `bench`/`codspeed` |
| `graph-bench data prepare --selection S` / `data key --selection S` | generate or download datasets, JRE, jars; print the cache key |
| `graph-bench list --selection S` | stable, sorted plan: one benchmark id per line |
| `graph-bench bench --selection S [--db a,b] [--grafeo PATH] --out DIR` | local and pre-release run: one worker process per database, writes `DIR/results.json` |
| `graph-bench codspeed --selection S --out DIR` | CI entry point: one `pytest benches/ --codspeed --db X` process per database, keeps `CODSPEED_PROFILE_FOLDER`, then writes `DIR/results.json` (answers, no timings) |
| `graph-bench report --format summary\|readme --in results.json` | summary block for grafeo's README / full section for graph-bench's README |

- Benchmark names: `benches/test_<category>.py::test_<benchmark>[<db>-<scale>]`, e.g. `benches/test_complex_queries.py::test_snb_is1[ladybug-small]`, and `benches/test_load.py::test_load_<dataset>[<db>-<scale>]`. Names must be unique across processes.
- Selections:
  - `push`: small scale, property-graph core categories, single-threaded.
  - `full`: every category and scale, single-threaded. All-cores runs come back with grafeo 0.6.4.
- Exit codes: 0 even when grafeo is slower or disagrees; non-zero only for harness errors.

### results.json

Top level:
- `schema_version`
- `graph_bench_commit`
- `grafeo` `{version, source, sha}`
- `machine` `{cpu, cores, os}`
- `selection`, `mode` (`bench` | `codspeed`), `started_at`
- `databases` `{name: {version, process, storage, threads_configurable, threads_note, java}}`

`entries[]`, one per (category, benchmark, database, scale, threads):
- `id`, `category`, `model` (`property_graph` | `sql_pgq` | `rdf`), `benchmark`, `database`, `scale`, `threads`
- `status`: `ok` | `failed` | `timeout` | `not_supported`
- `error`
- `timing` `{median_ns, p99_ns, min_ns, iterations}` (null in codspeed mode)
- `preflight_ns` (the untimed first run, wall-clocked; null when it did not complete)
- `load_ns`
- `answer`: `agree` | `disagree` | `unchecked`
- `reference`: `python` | `networkx`
- `answer_digest`
- `detail`

The grafeo side annotates entries that match `database == "grafeo" and answer == "disagree"`.

### Summary block (for grafeo's README)

```markdown
<!-- graph-bench:start -->
**graph-bench** · Grafeo <version> (<commit>) · <machine> · <date> · <N> embedded databases

| Category | DBs | Grafeo fastest | Grafeo median rank |
|---|---|---|---|
| Complex queries | 2 | 6 / 9 | 1 |

Single-threaded, <scale> scale; answers checked; wrong answers not ranked.
Full results, with every database named: https://github.com/GrafeoDB/graph-bench#results
<!-- graph-bench:end -->
```

grafeo's README never names other graph databases unless the user decides on that specific mention (user decision, 2026-10-30). So the block holds counts only; graph-bench's README and results.json name every database.

- One row per matrix category where Grafeo has at least one benchmark, in the matrix's category order.
- "DBs" counts the databases with at least one correct result in that row.
- "Grafeo fastest" is `k / n`: n is the number of benchmarks where Grafeo's answer agreed, ⏱ included (a load, which has no answer check, counts when it completed), and k the number where Grafeo has the lowest median time.
- "Median rank" is Grafeo's median position over those benchmarks.
- When Grafeo itself has a marker, a line under the table names the benchmark and the marker, e.g. `Grafeo: ≠ on snb_ic6`, so Grafeo's own gaps are never hidden. The legend line `⊘ not available · ✗ failing · ≠ wrong answer · ⏱ beyond 10x the top-3 average` is added only when such a line exists.

## CI and publishing

- **graph-bench workflow:** runs on `codspeed-macro` runners in wall-time mode, on push, on a schedule and on demand.
- **Separate CodSpeed project:** graph-bench's CodSpeed results are their own project, owned by the graph-bench repo. They sit next to grafeo's existing CodSpeed benchmarks, which keep finding grafeo regressions on releases, and never replace or feed them.
- **Grafeo build:** the workflow measures grafeo main built from source. This is the user's choice: main only, not the PyPI release.
- **graph-bench README:** a scheduled `full` run regenerates its results section and opens a bot PR.
- **grafeo README:** at each grafeo release (from 0.6.1), the pre-release checklist runs `setup --grafeo <wheel>`, `bench --selection full` and `report --format summary` locally, and commits the block on grafeo's release branch. No automation writes to grafeo.
- **Workflow switch-on:** the grafeo-side workflow trigger is gone; the graph-bench workflow is enabled once the adapters and the answer check have settled.

## Delivery order

1. **Foundation** (plan `2026-10-30-embedded-foundation.md`): legacy move, datasets, answer check with preflight and the 10x budget, per-process runner, results.json, Grafeo + Ladybug on Load, Complex queries, Hops (IC13) and Algorithms (BFS, WCC, PR), `setup`, `report` (matrix + summary), pytest-codspeed entry point, CI workflow. grafeo-31 is told when this works locally.
2. **Core graph categories:** Batch writes, Scans, Hops, Pattern matching (ported from the legacy suites, rewritten as checked queries).
3. **In-process Python adapters:** TuringDB embedded, FalkorDB (falkordblite), NeuG, HelixDB.
4. **JVM adapters:** ArcadeDB, Neo4j Community via JPype, JRE and jar fetching, memory cap.
5. **Vector search and Hybrid.**
6. **Deriva workload** (after the deriva session's input).
6b. **Crash and restore (category `crash_restore`):** crash and reopen tests run in a child process that the supervisor kills hard, with the reopen checked in a fresh process.
7. **SQL/PGQ** with DuckDB + DuckPGQ.
8. **RDF/SPARQL** with Oxigraph and BSBM.
9. **Project-published suites.**
10. **Remaining:** Graphalytics algorithms (CDLP, LCC, SSSP, Louvain, betweenness), `medium`/`large` scales.
12. **Grafeo loader:** when grafeo's native typed COPY-style loader lands (0.6.1, API from grafeo-28), Grafeo's Load switches from the batch API to it; until then the batch API stays (user decision, 2026-10-30).
13. **`release` selection with wiki-Talk** and LDBC validation references, before the 0.6.1 release run.
11. **With grafeo 0.6.4:** all-cores runs and Grafeo's real thread setting.
