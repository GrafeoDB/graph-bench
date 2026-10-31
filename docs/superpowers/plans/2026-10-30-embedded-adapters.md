# Embedded adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** All ten embedded databases run in `graph-bench bench --selection full` with every case either measured with a checked answer or carrying an honest marker, on Linux CI and locally.

**Architecture:** One adapter module per database under `graph_bench/databases/`, registered in `graph_bench/databases/__init__.py`, following the Grafeo and LadybugDB adapters. The harness becomes model-aware (property graph, SQL/PGQ, RDF) so DuckDB+DuckPGQ and Oxigraph take part through their own data model, with Grafeo in all three. JVM databases get a shared JVM/fetch helper.

**Tech Stack:** Python 3.12+, uv, pytest, pytest-codspeed; neug, falkordblite, turingdb, helix-db + helix-db-embedded, duckdb (+ duckpgq extension), pyoxigraph, arcadedb-embedded, JPype1 + Neo4j Community tarball + Temurin JRE.

**Spec:** `docs/superpowers/specs/2026-10-30-embedded-comparison-design.md`

**Research (read the section for your database before starting):**
`.superpowers/sdd/2026-10-30-embedded-adapters/research-a-report.md` (FalkorDB, NeuG, TuringDB),
`research-b-report.md` (HelixDB, DuckDB+DuckPGQ, Oxigraph, Grafeo SPARQL and SQL/PGQ, BSBM),
`research-c-report.md` (ArcadeDB, Neo4j via JPype).

## Global Constraints

- Latest stable release of every database, no pinning back, no nightlies or dev builds (user, 2026-10-30). A broken latest release shows its marker and is recorded in `docs/known-disagreements.md` with the upstream issue.
- Adapters use only documented paths, each cited with its docs URL in a code comment. Documented, fair fixes are allowed; nothing that cheats (precomputed answers, cross-iteration caching, skipped work, unsafe settings only one database gets). Grafeo problems go to the grafeo side (grafeo-31), never worked around.
- No em-dashes or en-dashes anywhere (code, comments, docs). Prefer uv over pip in docs.
- Shared query texts stay in their standard form; a per-dialect override only where the database's dialect requires it, with the reason in a comment. A database that cannot run the standard form gets a marker, not a weaker query.
- Storage in-memory where supported, else on disk in the instance's `workdir`, reported as `on-disk`. Single-threaded where configurable; otherwise `threads_configurable=False` with a note.
- Each adapter test (`tests/test_db_<name>.py`) runs the push selection for that database and requires agreement except allowlisted ids, like `tests/test_db_ladybug.py`.
- Linux-only wheels (falkordblite, turingdb) get `sys_platform == 'linux'` markers in pyproject extras; their tests skip where the package is missing. They must be verified on Linux (Docker: `docker --context desktop-linux`, image `ubuntu:24.04` or `python:3.12` with glibc >= 2.38 for falkordblite).
- Verification for every task: `uv run --no-sync pytest tests -q` (gate 88%), `uvx ruff@0.17.1 check .`, `uvx ruff@0.17.1 format --check .`. Never install into or sync the project venv without the controller's go-ahead; install new database packages with `uv pip install <pkg>` into the project venv only when the brief says so.
- Commits: short subject plus functional bullets, never a Co-Authored-By or Claude attribution.

## Review Focus

1. A database whose import or extension fails at runtime: every case shows ✗ with the database's own error, nothing crashes the supervisor.
2. A dialect override that silently changes the query's meaning (different filters, bounds, ordering): reviewers compare each override with the shared text line by line.
3. Load paths that skip work the others do (indexes built lazily outside the load clock, data loaded from a cache): the load clock covers everything until the data is queryable.
4. JVM processes: the JVM starts outside the clocks, gets the same memory cap, and one database per process.
5. Answers converted from foreign result types (Java boxed values, pandas, RDF literals) must compare equal to the references: ints stay ints, floats stay floats.

---

### Task 1: Harness residuals

Brief: `.superpowers/sdd/2026-10-30-embedded-adapters/task-1-brief.md` (in-run cutoff window, flaky tests, load count check, IC13 unreachable binding, small items).

### Task 2: NeuG adapter

**Files:** Create `graph_bench/databases/neug.py`, `tests/test_db_neug.py`. Modify `graph_bench/databases/__init__.py`, `graph_bench/suites/snb.py` (a `neug` dialect reusing the LadybugDB texts where research-a shows NeuG needs them), `pyproject.toml` (extra `neug`, added to `embedded`), `uv.lock`, README database table, `docs/known-disagreements.md` as needed.

- `name="neug"`, `display_name="NeuG"`, `models={"property_graph"}`, `dialect="neug"`, in-process, in-memory, `max_thread_num=1`.
- Load: DDL + `COPY ... (header=true, delim=",")` per research-a. Algorithms through the bundled GDS extension (`INSTALL gds` in `prepare_environment`, `LOAD gds` on first use like LadybugDB's ALGO). `counts()` implemented.
- Install `neug` into the venv with `uv pip install neug` (allowed for this task) and add it to the lock.

### Task 3: FalkorDB adapter (falkordblite, local process)

**Files:** Create `graph_bench/databases/falkordb.py`, `tests/test_db_falkordb.py`; modify registry, pyproject (extra `falkordb` with `sys_platform == 'linux'`), lock, README, known-disagreements.

- `process="local process"`, `threads_configurable=False` (note: thread count fixed at module load by the wrapper). Load: `CREATE INDEX` on `:Label(id)` per label, then UNWIND batches from the typed CSV rows, in the load clock. Queries: shared Cypher; IC13 is a marker if FalkorDB cannot run the standard text (research-a: undirected shortestPath and OPTIONAL MATCH shortestPath unsupported), recorded in known-disagreements, no rewrite. Algorithms: `algo.WCC`, `algo.pageRank` (record its fixed parameters); BFS via the documented form in research-a if it gives levels, otherwise ⊘.
- Verify on Linux in Docker (the wheel needs glibc 2.38); tests skip on Windows.

### Task 4: Multi-model harness (SQL/PGQ and RDF)

**Files:** `graph_bench/databases/base.py`, `graph_bench/runner/worker.py`, `graph_bench/suites/` (new `snb_sql.py`, `snb_rdf.py` or equivalent), `graph_bench/datasets/` (RDF mapping module), `graph_bench/databases/grafeo.py`, catalog/selections, `benches/` (new `test_sql_pgq.py`, `test_rdf.py`), tests.

- A database declares one dialect per model it supports (e.g. Grafeo: cypher for property_graph, sql_pgq for SQL/PGQ, sparql for RDF); `query()` gets the model so a multi-model adapter routes to `execute`, `execute_sql` or `execute_sparql`. Keep single-model adapters unchanged in shape (default mapping from their `models` and `dialect`).
- Loads are per (dataset, model) where the model needs its own store: the RDF load maps the SNB CSVs to triples with the mapping in research-b section "Oxigraph" (node IRI `base/<Label>/<id>`, `rdf:type`, one triple per non-empty column incl. `v:id`, edges `<src> v:<TYPE> <dst>`), implemented once in `graph_bench/datasets/rdf.py` and shared by all RDF loaders. SQL/PGQ uses the property graph store on Grafeo.
- Suites: SQL/PGQ category with SNB IS1, IS4, IS6, IC13 in standard SQL/PGQ (ISO 9075-16 forms; per-engine text only where research-b shows syntax differs AND both forms are standard); RDF category with SNB IS1, IS4, IS6 in SPARQL (inline `VALUES` first in the group). Reference answers are the existing SNB references (same data, same answers).
- Grafeo runs both categories; its gaps show markers (no workarounds). Grafeo's RDF load uses `INSERT DATA` chunks until grafeo-31 sends the Python RDF loader API (ledger note).
- Selections: `push` and `full` include `sql_pgq` and `rdf`.

### Task 5: DuckDB + DuckPGQ adapter

**Files:** `graph_bench/databases/duckdb.py`, `tests/test_db_duckdb.py`, registry, pyproject (extra `duckdb`), lock, README, known-disagreements.

- `models={"sql_pgq"}`, in-process, in-memory, `SET threads=1`. `INSTALL duckpgq FROM community` in `prepare_environment`; `LOAD duckpgq` and `SET active_grammar_extensions=['duckpgq']` per connection. Load: typed `CREATE TABLE` + `COPY`, the derived `Message`/`REPLY_OF` union tables research-b needed (documented reason: unique labels), `CREATE PROPERTY GRAPH`, all in the load clock. Algorithms: `pagerank`, `weakly_connected_component` table functions; BFS via `ANY SHORTEST` query; check PageRank against `LDBC_EPS` and record the outcome honestly.

### Task 6: Oxigraph adapter

**Files:** `graph_bench/databases/oxigraph.py`, `tests/test_db_oxigraph.py`, registry, pyproject (extra `oxigraph`), lock, README.

- `models={"rdf"}`, in-process `Store()`, load with `bulk_extend` of the shared RDF mapping, queries via `store.query`, terms converted by datatype. Send the RDF mapping file path to grafeo-31 when this lands.

### Task 7: TuringDB adapter

**Files:** `graph_bench/databases/turingdb.py`, `tests/test_db_turingdb.py`, registry, pyproject (extra `turingdb`, linux marker), lock, README, known-disagreements.

- Embedded `TuringDB(type="embedded", data_dir=workdir)`, on-disk, threads not configurable. Load: CSV to parquet conversion (pyarrow) plus `LOAD PARQUET`, all in the load clock. Parameters are inlined as literals by a quoting function (TuringDB has no value parameters, research-a), tested against injection-breaking characters. Overrides only where TuringDB's dialect requires (record each). No native algorithms: ⊘. Verify in Docker.

### Task 8: HelixDB adapter

**Files:** `graph_bench/databases/helixdb.py`, `tests/test_db_helixdb.py`, registry, pyproject (extra `helixdb`), lock, README.

- `Client.embedded(InMemory(...))`, in-process. HelixQL has no text form in the SDK: the adapter maps each query id to a builder function written with the documented Python DSL that computes the same result as the shared text (dialect `helix`; the suite's `text["helix"]` names the builder). Unsupported queries are ⊘. Load: `write_batch` chunks per research-b. Algorithms: BFS through `NativeGraph` if documented, WCC/PR ⊘.

### Task 9: JVM helper and ArcadeDB adapter

**Files:** `graph_bench/databases/jvm.py` (shared: memory cap, JVM start outside clocks, fetch-and-verify helper for downloads into the data cache with sha256), `graph_bench/databases/arcadedb.py`, tests, registry, pyproject (extra `arcadedb`), lock, README, FAIRNESS (JVM memory cap now implemented).

- On-disk in `workdir`, single-thread JVM settings from research-c, heap cap equal for all JVM databases. Load via `graph_batch` (the Graph Importer mangles quoted CSV values: record it in known-disagreements). Shared Cypher. Algorithms `algo.bfs` (add source at depth 0), `algo.wcc`, `algo.pagerank`.

### Task 10: Neo4j Community adapter (JPype)

**Files:** `graph_bench/databases/neo4j.py`, tests, registry, pyproject (extra `neo4j-embedded` with JPype1), lock, README, known-disagreements.

- `prepare_environment`: fetch the Community tarball and Temurin JRE into the data cache with sha256 checks (research-c URLs), outside every clock. Embedded `DatabaseManagementServiceBuilder` on a temp dir, JVM flags from Neo4j's default config. Load with `neo4j-admin database import` (provisional decision 7 in the ledger's decisions.md), timed. Shared Cypher; IC13 same-node failure recorded honestly unless Neo4j documents a setting. GDS algorithms (WCC, PageRank with documented parameters, BFS via unweighted single-source shortest paths), all through documented procedures.

### Task 11: Full run on all ten

- CI (`.github/workflows/ci.yml`) installs every extra on Linux and runs the adapter tests; `bench.yml` unchanged except extras. Run `graph-bench bench --selection full` with all ten databases on Linux (Docker, ubuntu:24.04) and the eight available on Windows; fix every harness failure at its root; record every database failure as a marker with its cause. Push to release/0.3.0, confirm CI green, merge to main.
