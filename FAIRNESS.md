# Fairness

The rules every database in graph-bench is measured under. Every adapter, whether maintained here or contributed,
must follow them. Pull requests that improve an adapter are welcome when they do.

graph-bench compares **embedded** graph databases: databases that run inside the benchmark's process, or in a
local process the Python package starts for it. The server comparison is out of scope here; the pre-0.3.0
benchmark, servers included, is kept in [legacy/](legacy/README.md).

## The rules

1. **Each database runs in its own process.** A Python process can host only one JVM, and separate processes
   keep allocators and caches from interfering. `graph-bench bench` starts one worker per database;
   `graph-bench codspeed` starts one `pytest benches --codspeed` process per database.
2. **JVM databases get the same warm-up as the others**, and JVM startup is not timed in Load or queries; the
   Startup row times every database's cold start on purpose. Each JVM database starts its process's JVM once,
   before any timed load (`prepare_environment`), and one JVM database runs per process.
   Every JVM database gets the same maximum heap, `-Xmx4g` (`MAX_HEAP` in `graph_bench/databases/jvm.py`),
   checked after the JVM starts. Without a cap the JVM would take a quarter of the machine's RAM; 4 GiB is
   ArcadeDB's own default and leaves room on a 16 GB CI runner for off-heap memory and the harness. The native
   databases have no runtime memory cap to set.
3. **Every database runs single-threaded.** A database whose thread count can't be set is marked as such, and
   each database's `threads_note` in results.json says how the setting was applied:
   - Grafeo: `RAYON_NUM_THREADS=1` in its worker, note "single-threaded executor". Grafeo's query executor is
     single-threaded until parallel queries land in grafeo 0.6.4; only its rayon parts (vector batch,
     clustering, folds) are parallel, and the variable sets those to one thread.
   - LadybugDB: `max_num_threads=1`.
   - JVM databases (ArcadeDB, Neo4j Community): the JVM's own threads are capped too, `-XX:ParallelGCThreads=1`,
     `-XX:ConcGCThreads=1` (the default collector is kept) and `-XX:CICompilerCount=2`, the lowest HotSpot accepts
     with tiered compilation, which stays on (one C1 and one C2 compiler thread). The JVM's other service threads
     are not capped.
   - ArcadeDB: also its query, scan and async thread pools and the JVM's common fork-join pool set to 1 (the note
     lists them); its housekeeping threads are not capped.
   - Neo4j Community: Cypher runs each query on the calling thread (the parallel runtime is Enterprise only); GDS
     gets `concurrency: 1` and `readConcurrency: 1`, `neo4j-admin database import` gets `--threads=1` and the
     JVM caps above. Its background threads (page cache flushing, checkpointing, index population) have no
     documented setting and are not capped.

   **Planned with grafeo 0.6.4:** the all-cores runs come back to the `full` selection, reported as a separate
   set of tables, and Grafeo's adapter switches from `RAYON_NUM_THREADS` to the real thread setting 0.6.4
   brings. Selections and benchmark ids already carry a threads value (an `-allcores` suffix), so this is a
   data change, not a redesign.
4. **Storage is in-memory where the database supports it.** A disk-only database runs on disk and is labelled
   so. Loading is measured separately from queries: each database loads the same CSV files with its own
   documented bulk-load path (see [Loading](#loading)).
5. **Answers are checked.** Each query's answer is compared with a Python reference computed from the generated
   data (NetworkX for algorithms). A database with a different answer is marked `disagree` (≠) and left out of
   the ranking.
6. **Queries return scalar values only** (no node or edge objects) and order fully, with tie-breakers, whenever
   they use `LIMIT`.
7. **PageRank is compared at convergence** (damping 0.85), with the LDBC relative epsilon of 0.0001.
8. **The time budget per benchmark is 10x the average median of the 3 fastest *other* databases with correct
   answers** (the average over those available if fewer than 3; no budget when no other database has a correct
   result). This final rule, on medians, decides the ⏱ marker.
   - **During a run, a runaway guard:** databases run one after another. Once at least 3 earlier databases have
     correct times, a later database's preflight, its first (cold) run, is cut off at 30x the average of their
     3 fastest preflights, and never below 10 s (a cold first run, such as a JVM compiling a new query, can take
     tens of milliseconds where the timed median is within budget): the supervisor stops the worker as soon as the preflight's database call runs past
     that budget, and the case becomes ⏱ ("too slow: cut off at ..."). Only the database call is timed against
     the cutoff; graph-bench's own work around it (loading the reference, checking the answer) is not. The
     cutoff only stops runaway cases early; because it measures cold first runs, it is set well above the final
     rule. It depends on which databases ran earlier, so a ⏱ from the cutoff can depend on the run order. Before 3 earlier results exist, only a hard
     cap applies (`--case-timeout`, default 600 s, measured as silence: the worker reports progress after every
     run).
   - **After the run:** the 10x rule is applied over all results, each entry against the other databases. Any
     benchmark beyond its budget becomes ⏱. The run order never decides a ⏱ from this final rule (unlike the
     in-run cutoff above).

## The preflight

Each benchmark first runs once, untimed: the preflight. The answer check and the time budget are applied there,
and only a benchmark that passes the preflight is timed, so a wrong or too slow result never reaches CodSpeed as
a speed. The preflight also counts as the first warm-up run. Warm-up runs, timed runs and load repetitions are
set per selection (`graph_bench/suites/catalog.py`) and are the same for every database.

A ⏱ assigned only by the final budget rule comes after CodSpeed has measured the case: CodSpeed then shows it as
measured, while results.json and the matrix show ⏱. A ⏱ from the in-run cutoff is never measured.

## What is timed

- **Queries:** the database call plus materializing its result into Python rows. This is identical for every
  database. graph-bench's canonicalization of the rows and the answer check run outside the clock.
- **Loads:** timed separately, each into a fresh in-memory instance, so no load benefits from an earlier one.
  Loads are count-checked: right after each timed load, outside the clock, the adapter counts the nodes and
  edges loaded through the database's own query interface (`counts()`, e.g. a Cypher count over all nodes and
  over all relationships), and graph-bench compares them with the dataset's counts. A mismatch makes the load ≠
  (wrong answer), e.g. "loaded 4177 nodes, 37000 edges; expected 4177 nodes, 37860 edges"; a database without
  `counts()` leaves its load unchecked. The budget rule (rule 8) does not apply to loads. In
  `graph-bench codspeed`, the count check runs after CodSpeed has timed the load, so CodSpeed shows a ≠ load as
  measured, while results.json and the matrix show ≠.
- **Load (query language):** the same datasets loaded through each database's query language (for example
  `LOAD CSV` or `COPY ... FROM`), each into a fresh instance, timed around that load alone. Each instance is
  count-checked like Load and then answers one content probe, outside the clock: a catalog query whose answer
  depends on the loaded text values (a mismatch is ≠, "content probe <query>: ..."; a database that cannot run
  the probe query leaves the load unchecked, "content probe <query> not run: ..."). These instances are closed
  afterwards and never used by the queries. The row runs after its dataset's queries, and a failure in it fails
  only its own entry. The budget rule does not apply.
- **Startup:** a cold start, in a fresh process per run (`graph_bench/runner/startup.py`). The clock starts after
  graph-bench's own modules are imported and covers the driver's import, opening an empty database in a fresh
  directory (starting whatever the process lacks: a JVM, an engine, a server process) and one probe query on it,
  whose answer is checked (no nodes, `SELECT 1`, or no triples); closing is not timed, nor is the interpreter's own
  start, the same for every database. Downloads, archive extraction, checksum checks and extension installs run
  before, untimed (`provision()`), and never in the child. The entry is the median over `startup_iterations`
  processes (3 in `push`, 5 in `full`); the first is the preflight. It measures a process-cold start, not an
  OS-cache-cold one: the database's files are in the OS cache after the first run. In `graph-bench codspeed`,
  CodSpeed times the whole child processes, interpreter start included, while results.json records the
  in-process times. The budget rule does not apply.
- **Write cases** (Batch writes): each run is untimed setup (for a case whose setup builds the rows its write
  changes, such as a batch delete, followed by a read-back that must show them, else ✗ "setup did not reach its
  state: ..."), the timed write (one committed statement per batch), an untimed read-back compared with the
  reference rows after the write, an untimed teardown that undoes the write, and a read-back compared with the
  loaded state. Before the first run the read-back must show the loaded
  state, every run (preflight, warm-up and timed) is read back, and after the case the node and edge counts must
  equal those before it (else ✗, "the case left the graph changed: ..."). Write cases run after every read query
  of their dataset's instance. In `graph-bench codspeed` a run's read-back and teardown happen in the next round's
  setup, so CodSpeed times the writes alone; a later run whose read-back differs has then already been measured
  by CodSpeed, while results.json and the matrix show ≠.
- **Answer-only cases:** a case on an answer-check fixture (`catalog.ANSWER_ONLY_DATASETS`, today the multigraph
  fixture of the trail-semantics case `sem_trail_self_loop`) is run and checked like any query, but it has no time
  budget (in the run or at the end), is no yardstick for another database's budget and is never ranked: its
  detail cell is ✓ when the answer agrees (or its marker), it counts towards ✓ k/n and Grafeo's n, and it is
  nobody's fastest and no rank position. The fixture is loaded for its cases without a Load, Load (query language)
  or Memory and Disk row, so a load that fails or drops a self-loop or a parallel edge shows on the case itself.
- **A database panic** (a Rust panic that surfaces as pyo3's `PanicException`, which is not an `Exception`) is a
  ✗ with its message, "PanicException: ...", wherever the database call was (a query, an algorithm, a load, a
  close, the startup); the worker goes on with the next case instead of dying.

## Memory and disk

Memory and Disk are records, not timed benchmarks: they never reach CodSpeed and have no budget.

- **Where:** after a database's worker (or pytest) process has finished, the supervisor loads each dataset and
  store the database's models reach once more, each in a fresh child process with the worker's environment
  (`graph_bench/runner/resources.py`). The child first runs the database's untimed environment setup
  (`prepare_environment()`: a JVM or engine started and warmed), opens a fresh instance, then reads its own
  process tree's memory (`before`), runs the same load the Load row times, reads it again (`after`), checks the
  node and edge counts, closes the instance and measures the store's files.
- **What is read:** the resident set size (RSS, psutil) summed over the child and all its descendants. RSS is
  what psutil reads cheaply on Linux, Windows and macOS, often enough to sample. Its limits apply to every
  database alike: a page shared by several processes counts once per process; memory reserved but never touched
  (a JVM's unused heap) is not resident, so it does not count; memory a garbage collector touched stays resident
  until it is returned to the OS, so it counts. USS would be more exact but is too slow to read every 10 ms.
  No step forces a garbage collection, in any runtime: memory as a user sees it.
- **The process tree:** an importer the database starts during the load (Neo4j's `neo4j-admin` JVMs) is a child
  and counts, as the load's real cost. On Linux the child is a child subreaper, so a server that daemonizes
  (falkordblite's redis-server) is re-parented to it and stays in the tree.
- **Loaded** is after minus before: the memory the loaded data keeps (it may be negative, and is shown as it is).
  **Peak during load** is the highest reading of the whole tree, interpreter and started engine included, among
  `before`, `after` and the supervisor's samples, taken every 10 ms from outside the measured process, between
  the `before` and `after` times (one monotonic clock across processes). A load faster than one interval may
  have no sample in between; its peak is then the higher of `before` and `after`, and a short spike between two
  samples can be missed.
- **Disk** is the apparent size of the regular files under the store's paths (`Database.store_paths()`; symlinks
  are not followed), measured after `close()`, so each database's own shutdown has flushed and checkpointed. An
  in-memory store has none: "in memory".
- **Answers and markers:** a record carries its load's count check; a mismatch is ≠ in both tables, a load error
  ✗, a child silent for longer than the case timeout is killed (⏱), a load the database's models do not reach is
  ⊘. Lowest is bold.

## Loading

Load uses each database's **fastest documented bulk path, CSV parsing included**:

- Grafeo: the batch API (`batch_create_nodes_with_props`, `batch_create_edges`), with the CSVs parsed in Python.
  Grafeo's query-language `LOAD CSV` is 2x to 3x slower today; it will be tracked as its own category, Load
  (query language), so engine-side load fixes show up there. When Grafeo's native typed loader lands, it
  replaces the batch API for Load.
- LadybugDB: `COPY ... FROM` the CSV files.
- ArcadeDB: GraphBatch (`create_vertices`, then `new_edges` from an id to RID map), with the CSVs parsed in
  Python. Its faster Graph Importer does not honour CSV quotes, which the SNB text columns need (see
  [docs/known-disagreements.md](docs/known-disagreements.md)).
- Neo4j Community: `neo4j-admin database import full` (the offline bulk importer) on CSV copies with typed header
  lines, then the store opened embedded, a unique `id` constraint per label awaited online and, for
  Graphalytics, the GDS projection; all in the load clock, including the importer's JVM starts.

## Adapter rules

- **Documented paths only.** An adapter uses the database's documented, publicly shipped API and query language,
  never internal or undocumented calls.
- **The vendor-recommended API.** Each database is called the way its documentation recommends. For example,
  LadybugDB's binding deprecates a separate prepare plus execute ("The use of separate prepare + execute of
  queries is deprecated. Please using a single call to the execute() API instead."), so its adapter calls
  `execute()` once per binding. Its point lookups get ⏱ because execution alone is over 10x Grafeo's.
- **Doc-recommended settings are allowed when they keep the comparison like for like.** For example, LadybugDB's
  `var_length_extend_max_depth` is raised so unbounded variable-length paths keep their openCypher meaning.
- **Nothing that gives one database an advantage**: no precomputed answers, no caching of results between runs,
  no code paths that detect the benchmark, no plug-ins written for the benchmark.
- **Native algorithms only.** Algorithms use the database's own implementation, as a procedure or a query in its
  own language. graph-bench never falls back to NetworkX inside the comparison: a database without one gets ⊘.
  NetworkX is only the reference for the answer check.
- **A database's bugs are recorded, not worked around.** A wrong answer, a crash or a case that runs far too long
  stays in the results as it is, and is described in [docs/known-disagreements.md](docs/known-disagreements.md)
  with the data, the database's answer, the reference and the cause.

## Markers

| Marker | Meaning |
| --- | --- |
| time, fastest **bold** | completed, answer checked |
| ⊘ | not available: the database lacks the capability or the data model |
| ✗ | failing: an error or a crash |
| ≠ | wrong answer: fails the answer check, not ranked |
| ⏱ | too slow: beyond the budget of rule 8 |

How each marker shows in CodSpeed is in the [README](README.md#markers).

## Licensing and process kinds

- **Neo4j** is measured as Neo4j Community only: the Enterprise terms restrict publishing benchmarks. Neo4j
  Community is GPLv3; its algorithms use GDS from the Community tarball, whose jar carries GPLv3 (`LICENSE.txt`)
  and runs as the GDS Community Edition without a licence file. The tarball and the JRE are downloaded into the
  data cache, never redistributed.
- **FalkorDB** runs through `falkordblite`, which starts a Redis server subprocess; it is reported as a
  "local process", not in-process.

## Datasets

- Every database in a run gets the same data, generated from a fixed seed or downloaded, never committed.
- Each dataset is a directory of CSV files, one per node label and one per (edge type, source label, target
  label), with a `schema.json` of typed columns and the reference answers.
- Dates are integer epoch milliseconds, so date types never decide an answer.
- Graphalytics uses the SNB `KNOWS` graph stored in both directions, so directed algorithm implementations give
  the undirected answer.

## LDBC attribution

The SNB Interactive and Graphalytics workloads are **inspired by** specifications of the
[Linked Data Benchmark Council (LDBC)](https://ldbcouncil.org/). They are **not official LDBC Benchmarks** and
have not been audited by the LDBC Council; see the [README](README.md#ldbc-disclaimer). LDBC specifications are
used under the [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Contributing adapter changes

A pull request that changes an adapter must:

1. follow the rules above;
2. touch one adapter (changes to shared code go in a separate pull request);
3. keep the adapter's push agreement test passing (see [CONTRIBUTING.md](CONTRIBUTING.md));
4. say what changed and why, with a link to the database documentation it follows.

If in doubt whether a change is fair, open an issue first.
