# Upstream issue drafts

One Markdown file per database, written in GitHub issue style, holding the bugs and defects that graph-bench found
through its answer checking (wrong answers, parse errors on standard syntax, packaging gaps, a mangling importer,
intermittent failures). Performance-only items and documented design limits are left out; each file ends with a short
"Not filed" list.

**Nothing has been filed yet.** Every file is ready to file after review, once the benchmark results page is public.
Until then the drafts link to it as the placeholder `<results page link>`; replace it before filing. Each reproduction
is a self-contained script (no graph-bench code) that was run for real, and the "Actual" blocks are the pasted output.
Versions were the latest stable on PyPI on 2026-10-31.

| Database | Issue title | Status | Upstream |
| --- | --- | --- | --- |
| LadybugDB | [An OPTIONAL MATCH named path with no match is a partial path, not NULL](ladybugdb.md) | draft, reproduced on 0.22.1 | no duplicate found |
| LadybugDB | [ALGO PageRank does not redistribute dangling and isolated vertices](ladybugdb.md) | draft, held: reproduced on 0.19.1 only (ALGO does not load on 0.22.1) | no duplicate found |
| LadybugDB | [ALGO extension fails to load on Linux (libnetworkit) and Windows (OpenSSL DLLs)](ladybugdb.md) | comment-on-existing | https://github.com/LadybugDB/ladybug/issues/857 and https://github.com/LadybugDB/ladybug/issues/1190 |
| DuckDB + DuckPGQ | [Label disjunction `(m:Post\|Comment)` in GRAPH_TABLE is a parse error](duckdb-duckpgq.md) | draft, reproduced | no duplicate found |
| DuckDB + DuckPGQ | [SQL/PGQ IC13 (ANY SHORTEST over undirected KNOWS) does not finish on 10,000 persons](duckdb-duckpgq.md) | draft, reproduced (400 s without output at 10,000 persons) | no exact duplicate; related https://github.com/cwida/duckpgq-extension/issues/279 |
| DuckDB + DuckPGQ | [PageRank stops at a fixed tolerance of 1e-6](duckdb-duckpgq.md) | comment-on-existing | https://github.com/cwida/duckpgq-extension/issues/187 |
| ArcadeDB | [Graph Importer CSV reader does not honour quoted fields](arcadedb.md) | draft, reproduced | no duplicate found; related #7267, #8046 |
| HelixDB | [Intermittent "unsupported cascades plan" during write_batch loads](helixdb.md) | draft, NOT reproduced in isolation (load dependent) | no duplicate found |
| FalkorDB | [shortestPath only in WITH or RETURN and not undirected, so OPTIONAL MATCH shortestPath fails](falkordb.md) | draft, reproduced | no duplicate found |
| FalkorDB | [falkordblite has no wheel for glibc older than 2.38 (Ubuntu 22.04)](falkordb.md) | draft, packaging request, reproduced on arm64 | no duplicate found |
| TuringDB | [openCypher shortestPath pattern does not parse](turingdb.md) | draft, reproduced | no duplicate found |
| Neo4j | nothing to file | none | n/a |
| NeuG | `MATCH (n) RETURN count(n)` returns no row on a database with no node tables | draft | none |
| Oxigraph | nothing to file | none | n/a |
| Grafeo | [tracked in the grafeo repository](grafeo.md) | reference list only | n/a |
