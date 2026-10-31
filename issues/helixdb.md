# HelixDB: issues to file

Upstream: https://github.com/HelixDB/helix-db.
Latest stable on PyPI, checked 2026-10-31: `helix-db` 0.3.4 and `helix-db-embedded` 0.3.3.

One draft, with an honest caveat: it was not reproduced in isolation.

---

## 1. Intermittent planner error "unsupported cascades plan" during write_batch loads

**Status:** draft. File only if the maintainers want a report without a deterministic reproduction. **Upstream:** no duplicate found (searched `HelixDB/helix-db`: "cascades", "planner error", "write_batch"; the hits are unrelated).

### Environment

- `helix-db` 0.3.4 and `helix-db-embedded` 0.3.3 (PyPI, latest stable).
- Seen on Windows 11 Pro 10.0.26200 (Python 3.12) and on Linux in Docker (Ubuntu 24.04, Python 3.12).

### Minimal reproduction

There is none. This is the honest status: **not reproduced in isolation**, so no script is attached.

What was tried and did not trigger it:

- HelixDB alone through the benchmark worker, on Windows and Linux, all four dataset loads.
- Ten repeated loads of the Graphalytics dataset in a loop, under 16 busy CPU loops, with `PYTHONHASHSEED` 0 to 15.
- A clean sequential run with eight databases on Windows: every HelixDB load succeeded.

What triggered it: in both full benchmark runs that shared the machine with a second full run (two heavy processes at
once, CPU and memory contended), HelixDB failed during `load()`:

- Windows: the `snb-medium` load.
- Linux in Docker: both Graphalytics loads and the `snb-medium` load.

The batches sent are the same in every run (same CSV rows, same order, same chunk size), so the failure depends on
timing or resources, not on the input. The call is `write_batch` on the embedded client, with chunks of a few
thousand nodes or edges, executed one after the other from a single thread.

### Expected

A write batch either succeeds or reports a data error. A planner failure that depends on machine load, with the same
input, should not happen; if the optimizer cannot find a plan inside a time or resource budget it should fall back
to a default plan (or retry), not abort the write.

### Actual

```text
HelixError: embedded HelixDB error: message='Query error: planner error: unsupported cascades plan: selected optimizer result did not contain a best physical alternative'
```

Raised by a `write_batch` during the load; the load fails and every case that depends on it is reported as failed.

Hypothesis (unverified): the cascades optimizer has a time or step budget that expires under CPU contention and the
"no best physical alternative" result is then treated as an unsupported plan instead of a timeout. Useful to know:
is the optimizer budget wall-clock based, and is there a setting to make it deterministic?

### Found by

graph-bench (load step of the full benchmark selection), <results page link>.

---

## Not filed (by design or performance only)

- IS3, IS7 and IC6 cannot be expressed in the HelixDB DSL: documented design limit, reported as "not available".
- No BFS, WCC or PageRank: documented, reported as "not available".
- Loading goes through `write_batch` chunks, and slow cases under the 10x budget rule: performance only.
