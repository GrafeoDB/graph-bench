# Resources and core categories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every database's cold start, memory and disk use, query-language load and the four core graph categories (Batch writes, Scans, Hops, Pattern matching) are measured with checked answers for all ten embedded databases, in results.json and the README matrix.

**Architecture:** Two kinds of measurement. *Benchmarks* (Startup, Load (query language), the core categories) are catalog items run through the worker and the pytest-codspeed entry point like today's cases, with the same preflight, markers and budgets (Startup and both load rows exempt from the budget rule). *Resources* (Memory, Disk) are taken by the supervisor in separate short-lived processes it samples from outside; they go into a new top-level `resources` list in results.json and never reach CodSpeed. Write cases get untimed setup, read-back and teardown around each timed run and run last on their dataset's instance. Core read categories share openCypher texts with Python references; cheap standard SQL/PGQ and SPARQL twins share those references.

**Tech Stack:** Python 3.12+, uv, pytest, pytest-codspeed, psutil (already a dependency), the ten database packages from the previous plan.

**Spec:** `docs/superpowers/specs/2026-10-30-embedded-comparison-design.md` (matrix, fairness rules, results.json, delivery order item 2). User decisions: `.superpowers/sdd/2026-10-30-embedded-adapters/decisions.md` "Resolved by the user", items 7, 16, 17, 18.

**Builds on:** `docs/superpowers/plans/2026-10-30-embedded-adapters.md` (all ten adapters, model-aware harness, JVM helper).

## Decisions (answered by the user 2026-10-31: all four as recommended: twins in the core categories; TuringDB ⊘ and ArcadeDB run in Load (query language); Memory and Disk in push and full; Startup also in CodSpeed)

1. Placement of the SQL/PGQ and SPARQL twins of the core read benchmarks (Tasks 5 to 7; they share the Cypher item's references through `same_as`): (a) extra rows in the core category itself, so DuckDB and Oxigraph appear in Scans, Hops and Pattern matching (recommended); (b) rows in the existing SQL/PGQ and RDF/SPARQL categories, as the SNB IS twins are today; (c) no twins, DuckDB and Oxigraph ⊘ by model in every core category.
2. Load (query language) for the two doubtful loaders: TuringDB documents `LOAD CSV` "for small files only" (https://docs.turingdb.ai/import_data/csv; snb-small has 37,860 edges); ArcadeDB's documented query-language import (SQL `IMPORT DATABASE`) may be the Graph Importer that does not honour CSV quotes. (a) Run both, markers fall where they fall; (b) TuringDB ⊘ quoting its docs' limit, ArcadeDB run (recommended); (c) both ⊘.
3. Memory and Disk in the push selection: (a) in push and full, so every adapter test exercises them, at one extra load per dataset per database in a fresh process (recommended); (b) full only, keeping the CodSpeed CI job shorter.
4. Startup in CodSpeed: (a) a CodSpeed benchmark timing the whole child process (interpreter start included), while results.json records the child's own in-process time, so Grafeo startup regressions show and `graph-bench list` keeps matching the pytest ids (recommended); (b) results.json only, absent from CodSpeed.

## Global Constraints

- Latest stable release of every database, no pinning back, no nightlies or dev builds. A broken latest release shows its marker and is recorded in `docs/known-disagreements.md` with the upstream issue.
- Documented paths only, each cited with its docs URL in a code comment. Documented, fair fixes are allowed; nothing that cheats (precomputed answers, caching across iterations, skipped work, unsafe or relaxed-durability settings only one database gets). Grafeo problems go to the grafeo side (grafeo-31), never worked around.
- A database failure is recorded with its cause, never worked around; a case that cannot run the shared text gets a marker, not a weaker query.
- Shared texts stay standard (openCypher; SQL/PGQ per ruling R6; SPARQL 1.1 with the inline `VALUES` convention of `snb_rdf.py`). Per-dialect overrides only for pure dialect syntax that keeps the meaning, with the reason in a comment.
- Answers are always checked: queries against the Python reference; writes by read-back after every run; startup by its probe answer; loads (both rows) by node and edge counts plus the content probe for Load (query language); memory records by the count check of the load they measured.
- Single-threaded where configurable; children (startup, resources) inherit the database's `process_env(threads)`; otherwise `threads_configurable=False` with a note.
- In-memory where supported, else on disk in the instance's workdir, labelled `on-disk`.
- No em-dashes or en-dashes anywhere. Prefer uv over pip in docs.
- Commits: short subject plus functional bullets, never a Co-Authored-By or Claude attribution.
- Verification for every task: `uv run --no-sync pytest tests -q` (coverage gate 88%), `uvx ruff@0.17.1 check .`, `uvx ruff@0.17.1 format --check .`. Never install into or sync the project venv without the controller's go-ahead.
- Linux-only wheels (falkordblite, turingdb) are verified in Docker (`docker --context desktop-linux`, `ubuntu:24.04`; falkordblite needs glibc 2.38+). Keep `--out` container-local (the Windows bind mount gave ENODATA). Their tests skip where the package is missing.
- CI runs on the Graviton arm64 runner (`codspeed-macro-arm64-graviton-ubuntu-22-04`); a package that cannot install there shows ✗ ("not installed") in every new row too (startup, both loads, memory and disk records).

## Review Focus

1. Clock boundaries: the `provision()` / `open_empty()` split must leave every Load clock covering exactly what it covered before; the startup clock starts after graph-bench's own imports and before the driver import; downloads, hashing, extraction and extension installs never inside a clock.
2. Write cases that skip work or leave state behind: every timed run starts from a verified loaded state, the read-back is checked after every run, writes run after every read case, and a counts guard follows each write case.
3. Memory sampling that misses processes or time: FalkorDB's daemonized redis-server (child subreaper) and Neo4j's importer JVMs must be in the sampled tree; the peak window uses the shared monotonic clock; sampling only from outside the measured process.
4. Hop and pattern texts whose answer depends on match semantics (relationship isomorphism vs homomorphism): the explicit node-distinctness conditions make both semantics give the reference answer; every override and twin must keep them.
5. Load (query language) instances are never queried by cases; their counts, content probe and indexes must match what their Load row does; they get no algorithm projections.

---

### Task 1: Benchmark harness (Startup, Load (query language), write cases, new categories)

**Files:** Modify `graph_bench/types.py`, `graph_bench/suites/__init__.py`, `graph_bench/suites/catalog.py`, `graph_bench/datasets/cache.py`, `graph_bench/databases/base.py`, `graph_bench/runner/timing.py`, `graph_bench/runner/worker.py`, `graph_bench/runner/supervisor.py`, `graph_bench/results.py`, `graph_bench/report/ranking.py`, `graph_bench/report/readme.py`, `graph_bench/report/summary.py`, `graph_bench/cli.py` (`list`), `benches/_cases.py`, `benches/conftest.py`, `tests/fakes.py` and tests (catalog, worker, supervisor, codspeed, report, results, timing). Create `graph_bench/suites/startup.py`, `graph_bench/runner/startup.py`, `benches/test_startup.py`, `benches/test_load_ql.py`. Docs: `FAIRNESS.md` (rule 2: "JVM startup is not timed in Load or queries; the Startup row times every database's cold start on purpose"; "What is timed" gains Startup, write cases, Load (query language)), the spec (matrix rows Startup and Load (query language) membership, delivery order note), `CHANGELOG.md`.

**Interfaces (Produces):**
- `types.py`: `Category` gains `"startup"`, `"load_ql"`; `CATEGORY_ORDER = ("startup", "load", "load_ql", "batch_writes", "scans", "hops", "pattern_matching", "complex_queries", ...rest unchanged)`; `CATEGORY_NAMES` "Startup", "Load (query language)"; `LOAD_CATEGORIES: frozenset[Category] = {"load", "load_ql"}`; `BUDGET_EXEMPT: frozenset[Category] = {"startup", "load", "load_ql"}`.
- `suites/__init__.py`: `@dataclass(frozen=True) class Write(_Eligibility)` with `id, title, category, model, dataset, run: Mapping[str, str], check: Mapping[str, str], setup: Mapping[str, tuple[str, ...]], teardown: Mapping[str, tuple[str, ...]], ordered: bool, rel_eps: float`; each text map falls back to the model's standard dialect like `Query.text_for`; `reference_id` is `id`. `Item = Query | Algorithm | Write`.
- `suites/startup.py`: `PROBE: Mapping[str, str]` (`cypher`: `MATCH (n) RETURN count(n) AS nodes`; `sql_pgq`: `SELECT 1 AS one`; `sparql`: `SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }`; `helix`: builder name `startup_probe`); `EXPECTED: Mapping[Model, list[list[Any]]]` (property_graph `[[0]]`, sql_pgq `[[1]]`, rdf `[[0]]`); `primary_model(models) -> Model` (property_graph if served, else sql_pgq, else rdf).
- `catalog.py`: `STARTUP_SCALE = "empty"`; `startup_id(db, threads)` gives `benches/test_startup.py::test_startup[<db>-empty]` (`-allcores` when threads is None); `startup_ids(selection, dbs)`; `load_ql_benchmark(dataset)` (`load_ql_<dataset>`); `load_ql_id(dataset, db, scale, threads)` (`benches/test_load_ql.py::test_load_ql_<dataset>[<params>]`); `load_ql_ids(selection, dbs)` (graph-store datasets only); `LOAD_QL_PROBES: dict[tuple[str, Model], str]` (empty here, filled by Task 5). `Selection` gains `startup_iterations: int` (push 3, full 5) and `resources: bool` (both True per open decision 3). Push categories gain `startup`, `load_ql`, `batch_writes`, `scans`, `pattern_matching`. Each suite module exports `QUERIES` (or `WRITES`), `bindings(item_id, graph, n)` and `reference(item_id, graph, binding)`; `catalog.reference_source(item_id)` returns that module and `cache._write_references` dispatches through it. A write reference file holds `{"bindings", "answers": <after rows>, "before": <before rows>}`; `load_write_reference(path) -> tuple[bindings, after, before]`.
- `base.Database`: `@classmethod provision(cls) -> None` (downloads, extraction, sha256 checks, extension installs; never starts an engine; default no-op); default `prepare_environment()` calls `provision()`, overriding adapters call it first; `open_empty(self) -> None` (start what this process lacks: JVM, engine, server process; open an empty usable database in `self.workdir`; default raises `NotSupported("startup")`); `load_ql(self, dataset) -> None` (default `NotSupported("no query-language bulk loader")`); `write(self, model, text, params) -> list[tuple]` (run and commit; default `query_model`); `store_paths(self) -> list[Path]` (default `[]`; used by Task 2).
- `timing.measure(fn, *, warmup, iterations, on_run=None, before=None, after=None)`: `before()` before and `after(result)` after every call (warm-up and timed), untimed and outside the GC-disabled window.
- `runner/startup.py`: child `python -m graph_bench.runner.startup --db NAME --threads (1|all) --workdir DIR` prints one JSON line `{"elapsed_ns": int, "rows": [...]}` or `{"error": str, "not_supported": bool}`; parent `run_startups(db, threads, iterations, workdir, timeout_s) -> StartupOutcome(times_ns: list[int], status: Status, answer: AnswerVerdict, error: str | None, detail: str | None)`.

**Decisions:**
- Startup clock: the child imports `graph_bench.databases.base`, `jvm`, `datasets.schema`, `datasets.rdf`, `suites.startup` first, then `t0 = perf_counter_ns()`; timed: `get_database(name)` (driver import), `cls(workdir, threads)`, `open_empty()`, the probe through `query_model(primary_model, PROBE[dialect], {})`; the clock stops when the probe returns; `close()` untimed. Interpreter start excluded (the same for every database).
- Startup runs: the worker (or pytest session) calls `db_cls.provision()` first, untimed (a provisioning error fails the startup entry with "provisioning: ..."), then `startup_iterations` sequential children, each in a fresh `workdir/startup-<i>`, each inheriting the worker's environment. Entry: `timing` = median/p99/min/count over the children (both modes), `preflight_ns` = first child, `answer` = agree only when every child's rows equal `EXPECTED[primary_model]`, `reference` = "python". A child silent past `STARTUP_TIMEOUT_S` (120 s) is killed: `timeout`. `NotSupported` from `open_empty` gives ⊘. Startup runs before any load. It measures a process-cold start, not an OS-cache-cold one (FAIRNESS says so).
- Write cases: before the preflight, the `check` text must return the `before` reference, else ✗ "the graph is not in its loaded state before the case: ...". Each run (preflight and timed): `setup` (untimed), `run` (timed; one `write()` call per binding, each committed), `check` (untimed, compared with `after`), `teardown` (untimed), `check` again (untimed, compared with `before`). The preflight decides the verdict; a later run whose read-back differs makes the entry ≠ with detail "read-back after timed run <k> differs: ...". After the case, untimed, `counts()` must equal the dataset's counts, else ✗ "the case left the graph changed: <count detail>". Write cases sort after every read case of their dataset (worker groups and `benches/conftest._order`).
- CodSpeed for writes: `benchmark.pedantic(target, setup=..., rounds=selection.iterations)`; the check and teardown of round k run in the setup of round k+1 and after the last round (the pattern `run_load` already uses).
- Load (query language): runs after the dataset's cases, once the dataset's instance is released; `load_iterations` fresh instances, each timed around `load_ql()` alone; each count-checked like Load; then, when `LOAD_QL_PROBES` has (dataset, primary model), that item's preflight call runs untimed on it against its references (mismatch: ≠ "content probe <item>: ..."); closed, never used by cases. A failure fails only this entry, never cascading.
- Budgets and ranking: `apply_budget_rule` and the in-run `_budgets` skip `BUDGET_EXEMPT`; `_Meta` gets `kind: Literal["case", "load", "load_ql", "startup"]`, and the "load failed" cascade applies only to `kind == "load"`; `ranking.is_load` becomes `category in LOAD_CATEGORIES`; `speed()` uses `timing.median_ns` for startup in both modes. Titles: `startup` "Empty database"; `load_ql_<dataset>` the dataset name.
- Report: "### Startup (empty database, ms)"; Load (query language) in seconds like Load; overview rows come from `CATEGORY_ORDER`; the summary block gets Startup and Load (query language) (counting like Load in `_agreed`).
- Ids: `graph-bench list` prints the startup and load_ql ids; `test_codspeed_names_match_list` still passes.

**Tests (fakes, exact names):** `test_category_order_starts_with_startup_and_loads` (`CATEGORY_ORDER[:3] == ("startup", "load", "load_ql")`); `test_startup_id_format` (`benches/test_startup.py::test_startup[fake-empty]`); `test_measure_runs_before_and_after_outside_the_clock` (fake clock: `before`/`after` contribute 0 ns); `test_write_case_phase_order` (fake trace equals `[check, setup, run, check, teardown, check]` per run); `test_write_wrong_readback_is_disagree`; `test_write_case_leaving_state_fails_the_guard`; `test_write_cases_run_after_reads`; `test_startup_uses_fresh_processes` (N distinct child pids and workdirs); `test_startup_never_provisions_in_the_clock` (fake records `provision` in the parent only); `test_startup_wrong_probe_is_disagree`; `test_startup_without_open_empty_is_not_available`; `test_load_ql_probe_mismatch_is_disagree`; `test_load_ql_failure_does_not_fail_cases`; `test_budget_rule_exempts_startup_and_load_ql`; `test_codspeed_write_case_uses_pedantic_setup` (via `benchmark_spy`); `test_report_startup_in_ms_and_load_ql_in_s`.

### Task 2: Resource harness (Memory and Disk)

**Files:** Create `graph_bench/runner/sampler.py`, `graph_bench/runner/resources.py`, `tests/test_sampler.py`, `tests/test_resources.py`. Modify `graph_bench/runner/supervisor.py`, `graph_bench/results.py`, `graph_bench/report/readme.py`, `tests/fakes.py`, `FAIRNESS.md` (new section "Memory and disk"), the spec (results.json `resources`, Memory and Disk tables), `CHANGELOG.md`.

**Interfaces (Produces):**
- `sampler.py`: `SAMPLE_INTERVAL_S = 0.01`; `tree_rss(pid: int) -> tuple[int, int]` (sum of `psutil.Process.memory_info().rss` over the process and `children(recursive=True)`, and the process count; `NoSuchProcess`/`AccessDenied` per child skipped); `@dataclass Sample(t_ns: int, rss_bytes: int, processes: int)`; `class TreeSampler(pid, interval_s=SAMPLE_INTERVAL_S)` with `.start()`, `.stop() -> list[Sample]` (a thread in the supervisor; `t_ns` is `time.monotonic_ns()`); `set_child_subreaper() -> bool` (Linux `prctl(PR_SET_CHILD_SUBREAPER=36, 1)` via ctypes libc, man 2 prctl; False elsewhere); `dir_size(paths: Sequence[Path]) -> int` (apparent size, `st_size`, of regular files under the paths, symlinks not followed).
- `resources.py` (child): `python -m graph_bench.runner.resources --db X --selection S --dataset D --store (graph|rdf) --scale SC --threads (1|all) --workdir W --out F`: (1) `set_child_subreaper()`; (2) `prepare_environment()`; (3) fresh instance; (4) line `{"kind": "memory", "phase": "before", "t_ns", "rss_bytes", "processes"}` (the child measures its own tree); (5) `load_store(...)`; (6) the `"after"` line; (7) count check; (8) `paths = store_paths()`, `close()`, `dir_size(paths)` or None when `paths == []`; (9) line `{"kind": "resource", ...}`.
- `results.py`: `SCHEMA_VERSION = 2`; `@dataclass Resource(id, database, dataset, store, scale, threads, status, error, answer, detail, storage, baseline_bytes, after_bytes, footprint_bytes, peak_bytes, samples, sample_interval_ms, processes, disk_bytes)`; `Results.resources: list[Resource]`; `Machine` gains `memory_bytes` (`psutil.virtual_memory().total`); `from_dict` accepts schema 1 (empty resources, no `memory_bytes`). Resource id: `resources::<load_benchmark>[<db>-<scale>]`.

**Decisions:**
- Why psutil: already a dependency; cross-platform (Linux /proc, Windows GetProcessMemoryInfo, macOS); walks the process tree; cheap RSS reads for 10 ms sampling. Rejected: `getrusage(RUSAGE_CHILDREN)` (Unix only, waited children, no load window); Linux `VmHWM` + `clear_refs` (per process, Linux only); Windows Job Object peak (Windows only); cgroup v2 `memory.peak` (needs delegation the CI runner lacks). USS is more exact but reads smaps, too slow for sampling. RSS everywhere, its limits stated in FAIRNESS (shared pages counted per process; a JVM's reserved but untouched heap is not resident, so not counted; heap the GC touched stays resident).
- Where: the supervisor launches resource runs after the database's worker (or pytest) process finished, in both modes, one child per (dataset, store, scale) the database's models reach, per threads setting; other loads get ⊘ records, a database not installed ✗ records.
- Peak and footprint: peak = max(child `before`, child `after`, every sample with `before.t_ns <= t_ns <= after.t_ns`); footprint = after minus before (may be negative, shown as is); `baseline_bytes` = before (engine already started and warmed by `prepare_environment`); `samples` = samples in the window (0 allowed for a very fast load, peak then max(before, after)). No forced GC in any runtime: memory as a user sees it.
- Process tree: the subreaper makes a daemonized descendant (redislite's redis-server) a child of the resource process, so `children(recursive=True)` sees it; Neo4j's importer JVMs are ordinary children and count toward the peak (the load's real cost).
- Disk: measured after `close()`, so each database's own shutdown has flushed and checkpointed; `storage == "in-memory"` gives `disk_bytes = None`, shown as "in memory".
- Timeouts and failures: silence beyond `case_timeout_s` kills the child (`timeout`); a load error `failed`; a count mismatch answer `disagree`, ≠ in both tables.
- Report: per scale "### Memory (<scale> scale, MiB)" with rows "<dataset name>, loaded" and "<dataset name>, peak during load" (RDF store rows "(RDF)"), and "### Disk (<scale> scale, MiB)" one row per dataset and store; lowest bold; after the category tables, before the legend; not in the overview or summary block. In codspeed mode the tables are the same, and the README notes that CodSpeed holds no memory or disk data.

**Tests:** `test_tree_rss_counts_children` (a child holding 64 MiB with a grandchild holding 64 MiB: the tree is at least 128 MiB above a bare interpreter); `test_sampler_sees_a_transient_peak` (a child allocates 200 MiB for 300 ms and frees it: peak minus baseline at least 150 MiB); `test_monotonic_clock_is_shared_across_processes` (parent reading <= child's `monotonic_ns` <= parent reading); `test_subreaper_adopts_a_daemonized_grandchild` (Linux only, Docker: a double-forked process appears in the tree); fakes: `test_resource_record_per_load`, `test_fake_allocation_shows_in_footprint` (fake flag `memory:load_snb=<MiB>`), `test_disk_size_of_store_paths` (exact bytes), `test_in_memory_disk_is_none`, `test_resource_count_mismatch_is_disagree`, `test_resources_off_when_selection_disables`, `test_results_v1_still_reads`, `test_report_memory_and_disk_tables`.

### Task 3: Startup on all ten databases

**Files:** all ten adapters (`graph_bench/databases/*.py`), `graph_bench/databases/helix_queries.py` (builder `startup_probe`: count of all nodes), the adapter tests, `docs/known-disagreements.md`.

**Interfaces (Consumes):** `Database.provision`, `Database.open_empty`, `suites.startup.PROBE` (Task 1).

**Decisions:**
- `provision()` holds only what is untimed by nature: Neo4j `install()` (fetch, sha256, extraction); LadybugDB `INSTALL ALGO`; NeuG `INSTALL gds`; DuckDB `INSTALL duckpgq FROM community`; nothing for the others. `prepare_environment()` stays `provision()` plus today's untimed warm start.
- `open_empty()` is what each load does before its first data: Grafeo `GrafeoDB()`; LadybugDB `Database(max_num_threads)`, `Connection`, `var_length_extend_max_depth`; NeuG `Database`, `connect`; FalkorDB `FalkorDB(None, serverconfig={"save": ""})`, `select_graph`; TuringDB the engine start when this process has none plus a new graph made the documented way, `set_graph`; HelixDB `Client.embedded(InMemory(...))`; ArcadeDB `jvm.start_once`, `create_database(workdir/db)`; Neo4j `jvm.start_once`, `_builder(workdir).build().database("neo4j")` with the same builder settings (GDS plugin dir included); DuckDB `connect(":memory:")`, `SET threads`, `LOAD duckpgq`, `SET active_grammar_extensions`; Oxigraph `Store()`.
- A load may call `open_empty()` where its first step is the same; Neo4j's cannot (the importer needs an unopened store directory) and stays as is. No Load clock gains or loses work.
- `open_empty()` never fetches, hashes or installs; on a missing provisioned file it raises "not provisioned: <path>".

**Tests:** per adapter `test_<db>_startup_agrees` (push startup entry ok/agree, median > 0); `test_neo4j_open_empty_never_fetches` (`jvm.fetch` and `install` monkeypatched to raise in the child environment); `test_<db>_load_unchanged` (push load still agrees). FalkorDB and TuringDB in Docker. known-disagreements notes: Neo4j's start includes GDS registration (the benchmark's configuration), ArcadeDB's includes its polyglot engine init, the JVM thread caps apply to startup.

### Task 4: Memory and Disk on all ten databases

**Files:** `graph_bench/databases/arcadedb.py`, `neo4j.py`, `turingdb.py` (`store_paths`), `tests/test_db_*.py`, `docs/known-disagreements.md`, README methodology.

**Interfaces (Consumes):** `Database.store_paths`, the resources child and supervisor (Task 2).

**Decisions:** Store paths: ArcadeDB `workdir/db`; Neo4j `workdir/data` (databases and transaction logs; excludes `conf/`, `import/`, `import.report`); TuringDB the graph's own directory in the engine's `data_dir`, excluding the Parquet staging under `data/<graph>` (the implementer confirms the layout and cites it). In-memory databases report `[]` (Grafeo, LadybugDB, NeuG, FalkorDB with `save ""`, HelixDB, DuckDB, Oxigraph).

**Tests:** per adapter `test_<db>_resources_ok` (one ok/agree record per push load with `peak_bytes >= max(baseline_bytes, after_bytes)`; on-disk databases `disk_bytes > 0`, in-memory `disk_bytes is None`); `test_falkordb_redis_server_in_tree` (Linux, Docker: the after-load tree holds a process named `redis-server`, `processes >= 2`); `test_neo4j_peak_includes_importer` (`processes`, the largest process count seen during the load, at least 2).

### Task 5: Scans

**Files:** create `graph_bench/suites/scans.py`; modify `catalog.py` (ITEMS, `LOAD_QL_PROBES`), `helix_queries.py`, dialect overrides in `scans.py` where needed, `benches/test_scans.py`, `tests/test_references.py`, adapter tests' allowlists, `docs/known-disagreements.md`.

**Items** (SNB, property graph; ported from legacy `query.py` / `storage.py`):
- `scan_label` "Label scan": `MATCH (m:Comment) RETURN m.id AS id`; unordered; one empty binding.
- `scan_filter` "Property filter": `MATCH (m:Comment) WHERE m.creationDate >= $from AND m.creationDate < $to RETURN m.id AS id`; unordered; bindings: `selection.bindings` windows each over 10% of the comments by creationDate, spread evenly.
- `scan_aggregate` "Aggregation": `MATCH (p:Person) RETURN p.browserUsed AS browser, p.gender AS gender, count(*) AS persons, min(p.birthday) AS firstBirthday, max(p.creationDate) AS lastCreated ORDER BY browser, gender`; ordered; one empty binding.
- `scan_order_limit` "Order plus limit": `MATCH (m:Comment) WHERE m.length >= $minLength RETURN m.id AS id, m.content AS content, m.creationDate AS created ORDER BY created DESC, id ASC LIMIT 20`; ordered; bindings: the distinct comment lengths spread evenly. Its content strings carry commas and quotes, so it is the Load (query language) content probe: `LOAD_QL_PROBES[("snb", "property_graph")] = "scan_order_limit"`, `[("snb", "sql_pgq")] = "scan_order_limit_sql"`.

**Twins** (`<id>_sql`, `<id>_rdf`, `same_as=<id>`, titles "<title> (SQL/PGQ)" / "<title> (SPARQL)", placement per open decision 1): all four in SQL/PGQ (R6 graph-pattern WHERE, an edge variable on every edge pattern, `$name` parameters) and SPARQL (`VALUES` first).

**Databases:** every property-graph adapter; HelixDB builders where the DSL expresses the query, else a reason string (⊘).

**Tests:** `test_scan_references_on_a_hand_graph` (exact rows on a 5-node graph); each adapter's push agreement test.

### Task 6: Hops

**Files:** create `graph_bench/suites/hops.py`; modify `catalog.py`, `helix_queries.py`, `benches/test_hops.py`, tests, known-disagreements.

**Items** (KNOWS either direction; ported from legacy `traversal.py` hop_1, hop_2, bfs depth 3; `snb_ic13` stays as is):
- `hop_1` "1-hop": `MATCH (s:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN count(DISTINCT f.id) AS persons`.
- `hop_2` "2-hop": `MATCH (s:Person {id: $personId})-[:KNOWS]-(:Person)-[:KNOWS]-(f:Person) WHERE f.id <> $personId RETURN count(DISTINCT f.id) AS persons`.
- `hop_3` "3-hop": `MATCH (s:Person {id: $personId})-[:KNOWS]-(a:Person)-[:KNOWS]-(b:Person)-[:KNOWS]-(f:Person) WHERE b.id <> $personId AND f.id <> a.id AND f.id <> $personId RETURN count(DISTINCT f.id) AS persons`.
- `hop_var` "Variable length 1 to 3": `MATCH (s:Person {id: $personId})-[:KNOWS*1..3]-(f:Person) WHERE f.id <> $personId RETURN count(DISTINCT f.id) AS persons`.

**Decisions:** KNOWS pairs are unique and the data has no self-loops (the generator guarantees both; the reference asserts them); under that, the explicit conditions make relationship-isomorphic (openCypher) and homomorphic (walk) engines give the same set, and `*1..3` is the radius-3 ball in both semantics. The reference computes walks with explicit edge identities and these conditions. Bindings: `selection.bindings` persons with friends, spread evenly (hubs included). Twins: SQL/PGQ all four (`-[k:KNOWS]-`, `{1,3}` for `hop_var`); SPARQL `hop_1`, `hop_2`, `hop_3` as basic graph patterns with named intermediates and `(v:KNOWS|^v:KNOWS)`; no SPARQL `hop_var` (SPARQL 1.1 has no bounded repetition).

**Tests:** `test_hop3_reference_excludes_repeated_edges` (path graph a-b-c: `hop_3` from a is 0); `test_hop_var_equals_bfs_ball` (against NetworkX `single_source_shortest_path_length(cutoff=3)` on snb-small).

### Task 7: Pattern matching

**Files:** create `graph_bench/suites/patterns.py`; modify `catalog.py`, `helix_queries.py`; create `benches/test_pattern_matching.py`; tests, known-disagreements.

**Items** (ported from legacy `pattern.py` triangle_count and common_neighbors, plus the spec's cycles and joins):
- `pattern_triangle` "Triangles through a person": `MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) WHERE b.id < c.id RETURN count(*) AS triangles`.
- `pattern_square` "4-cycles through a person": `MATCH (a:Person {id: $personId})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(d:Person)-[:KNOWS]-(a) WHERE b.id < d.id AND c.id <> $personId RETURN count(*) AS cycles`.
- `pattern_common` "Common friends": `MATCH (a:Person {id: $person1Id})-[:KNOWS]-(c:Person)-[:KNOWS]-(b:Person {id: $person2Id}) RETURN count(DISTINCT c.id) AS common`.
- `pattern_join` "Multi-pattern join": `MATCH (p:Person)-[:HAS_INTEREST]->(t:Tag), (m:Post)-[:HAS_TAG]->(t), (m)-[:HAS_CREATOR]->(p) RETURN t.name AS tag, count(*) AS posts ORDER BY posts DESC, tag ASC`; ordered; one empty binding.

**Decisions:** every text binds pairwise distinct nodes, so isomorphic and homomorphic engines give the same counts. Bindings: triangles and squares use `selection.bindings` persons with friends whose KNOWS degree is at most the median degree, spread evenly (bounds hub enumeration, deterministic); common friends use pairs half the id-sorted list apart, as IC13. Twins: SQL/PGQ and SPARQL of all four (cyclic patterns repeat the variable; SPARQL basic graph patterns with FILTERs).

**Tests:** `test_square_reference_counts_each_cycle_once` (K4: 3 four-cycles through a vertex); `test_triangle_reference_on_k4` (3).

### Task 8: Load (query language)

**Files:** `graph_bench/databases/grafeo.py`, `neo4j.py`, `ladybug.py`, `neug.py`, `duckdb.py`, `arcadedb.py`, `turingdb.py`, `falkordb.py` (`load_ql` or a NotSupported reason), adapter tests, `FAIRNESS.md` (Loading), README methodology, known-disagreements (rewrite "Grafeo: LOAD CSV is 2x to 3x slower" into this row's note).

**Interfaces (Consumes):** `Database.load_ql`, load_ql entries, `LOAD_QL_PROBES` (Tasks 1, 5).

**Decisions (one fresh instance per load):**
- Grafeo (grafeo `docs/user-guide/cypher/mutations.md` "LOAD CSV" and "CREATE INDEX"): the same named graph as Load; per node table `LOAD CSV WITH HEADERS FROM '<abs path>' AS row CREATE (:<Label> {...})` with `toInteger`/`toFloat` per typed column (empty fields give null, the dataset contract); then `CREATE INDEX FOR (n:<Label>) ON (n.id)` per label; then per edge table `MATCH` both endpoints by `id` and `CREATE`.
- Neo4j: `open_empty()` with `GraphDatabaseSettings.load_csv_file_url_root` set to the dataset directory (research-c); `CREATE CONSTRAINT ... REQUIRE n.id IS UNIQUE` per label and `CALL db.awaitIndexes`; then `LOAD CSV WITH HEADERS FROM 'file:///<file>' AS row CALL (row) { ... } IN TRANSACTIONS` with the documented default batch size, nodes before edges, all through `executeTransactionally` (implicit transactions only); no GDS projection. The Load row stays `neo4j-admin import` (user decision 7).
- LadybugDB and NeuG already load through their query language (`COPY ... FROM`), DuckDB through SQL `COPY` + `CREATE PROPERTY GRAPH`: their `load_ql` runs the same statements without the algorithm projections; each entry is measured on its own; the README says so.
- ArcadeDB and TuringDB per open decision 2; the implementer cites ArcadeDB's documented query-language importer and runs it as documented.
- ⊘ with the reason recorded: FalkorDB (`LOAD CSV` reads only from the module's `IMPORT_FOLDER`, a load-time setting falkordblite does not expose, research-a); HelixDB (no query-language text in the SDK; its Cypher page lists no LOAD CSV); Oxigraph by the model gate (graph store only; SPARQL `LOAD` belongs with the RDF/SPARQL plan).
- Every Load (query language) instance gets the same id indexes as its Load.

**Tests:** per adapter `test_<db>_load_ql_agrees` (counts and content probe), and `test_<db>_load_ql_reason` for the ⊘ ones (exact reason text).

### Task 9: Batch writes, suite plus Grafeo, LadybugDB, NeuG, FalkorDB

**Files:** create `graph_bench/suites/writes.py`, `benches/test_batch_writes.py`; modify `catalog.py`, the four adapters (`write` where `query` does not commit), tests, known-disagreements.

**Items** (SNB, property graph only; ported from legacy `write.py` property_update and edge_add_existing, and `storage.py` node_insertion). Each run executes 10 batches of 100 rows (`UNWIND $rows AS r ...`), one committed statement per batch, at every scale and selection; batches' person sets are disjoint.
- `write_insert` "Batch insert": run `UNWIND $rows AS r MATCH (f:Person {id: r.friend}) CREATE (:Person {<all Person columns from r>})-[:KNOWS {creationDate: r.since}]->(f)`; check `UNWIND $ids AS id MATCH (p:Person {id: id})-[k:KNOWS]->(f:Person) RETURN p.id AS id, p.firstName AS firstName, p.birthday AS birthday, f.id AS friendId, k.creationDate AS since`; teardown `UNWIND $ids AS id MATCH (p:Person {id: id}) DETACH DELETE p`; before `[]`.
- `write_update` "Batch update": run `UNWIND $rows AS r MATCH (p:Person {id: r.id}) SET p.browserUsed = r.browserUsed, p.locationIP = r.locationIP`; check returns id, browserUsed, locationIP; teardown the same update with `$original`.
- `write_delete` "Batch delete": setup the insert statement on its own ids; run `UNWIND $ids AS id MATCH (p:Person {id: id}) DETACH DELETE p`; check `UNWIND $ids AS id MATCH (p:Person {id: id}) RETURN p.id AS id` (before 100 rows, after `[]`); the counts guard covers the removed edges.
- `write_upsert` "Batch upsert": each batch 50 existing persons plus 50 new ids; run `UNWIND $rows AS r MERGE (p:Person {id: r.id}) SET p.browserUsed = r.browserUsed`; check returns id, browserUsed; teardown `DETACH DELETE` of the new ids, then restore `$original`.

**Binding rules:** new ids `winsert_<b>_<i>`, `wdelete_<b>_<i>`, `wupsert_<b>_<i>`, never in the data; every new value differs from the stored one (browserUsed the next browser after the original, cyclically; locationIP `10.<op>.<b>.<i>`); update and upsert targets are consecutive slices of the id-sorted persons; each new row's `friend` is spread over persons with friends.

**Decisions:** 100 rows is twice legacy `write.py`'s 50 single updates and fits 1,000 disjoint updates into snb-small's 1,001 persons. Writes on on-disk databases include their durable commit (reported in known-disagreements as an on-disk disadvantage; no durability setting is relaxed). DuckDB and Oxigraph ⊘ by model (SQL/PGQ defines no graph writes; SPARQL Update twins wait for the BSBM update profile).

**Tests:** `test_write_references_before_and_after` (hand graph, exact rows); `test_write_bindings_disjoint_and_new`; per adapter, push agreement with every write case ok/agree or allowlisted.

### Task 10: Batch writes on ArcadeDB, Neo4j, TuringDB, HelixDB

**Files:** `graph_bench/databases/arcadedb.py`, `neo4j.py`, `turingdb.py`, `helixdb.py`, `helix_queries.py`, adapter tests, known-disagreements.

**Decisions:** ArcadeDB `write()` runs the openCypher text through `db.command` inside `db.transaction()` (the documented write path). Neo4j `write()` is `_run(..., commit=True)`; today's `query()` never commits, so a write through it would roll back. TuringDB `write()` follows the documented change workflow (new change, writes, commit, submit) inside the timed call; `literal()` gains Cypher list and map literals (tested against injection characters like strings); an unsupported clause (UNWIND, MERGE, DETACH DELETE) is ✗ with TuringDB's own error, recorded. HelixDB builders `write_<op>:<phase>` using the documented DSL write steps (add_n, add_e; update, drop, upsert as documented); a step the DSL lacks maps to a reason string (⊘). FalkorDB and TuringDB verified in Docker.

### Task 11: Full run on all ten and known disagreements

- `graph-bench bench --selection full` with all ten in Docker (`ubuntu:24.04`, `--out` container-local) and the eight available on Windows, with nothing else running on the host; fix every harness failure at its root; record every database failure in the new rows as a marker with its cause.
- known-disagreements: one section per non-agreeing entry in the existing format (`## <Database>: <what happens> (...)`, the `- id: <db>/<benchmark>` allowlist line only for real disagreements, query, dataset, binding, the database's answer, the reference, cause engine vs adapter). Plus the Load (query language) notes, startup notes (process-cold, JVM caps, Neo4j with GDS), memory notes (RSS limits, sampling interval, Neo4j's importer peak). Drop the "Grafeo: medium-scale RDF" section if 0.6.1 is on PyPI (keep its IC13 part).
- README methodology (new rows, Memory and Disk tables) and CHANGELOG; Grafeo findings in the new rows go to grafeo-31.
- CI: the controller pushes and confirms CI green on the Graviton runner, with FalkorDB ✗ "not installed" in every new row and every other installable database present.
