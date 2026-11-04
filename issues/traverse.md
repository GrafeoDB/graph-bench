# Traverse: issues to file

Upstream: https://github.com/truespar/traverse
Latest stable on PyPI, checked 2026-11-04: `traverse-embedded` 0.9.0.

One draft.

---

## 1. `shortestPath` is refused under `OPTIONAL MATCH`

**Status:** draft, reproduced. **Upstream:** no duplicate found (the repository had no open issues on 2026-11-04).

### Environment

- `traverse-embedded` 0.9.0, PyPI, latest stable.
- Windows 11, Python 3.13.0.

### Minimal reproduction

```python
import traverse, tempfile, os
d=tempfile.mkdtemp(); db=traverse.open(os.path.join(d,'p.tvdb'))
db.execute("CREATE (:P {id:1})-[:K]->(:P {id:2}), (:P {id:3})")
for t in ["MATCH (a:P {id:1}),(b:P {id:2}) MATCH p=shortestPath((a)-[:K*]-(b)) RETURN length(p) AS l",
          "MATCH (a:P {id:1}),(b:P {id:2}) OPTIONAL MATCH p=shortestPath((a)-[:K*]-(b)) RETURN length(p) AS l",
          "MATCH (a:P {id:1}),(b:P {id:3}) OPTIONAL MATCH p=shortestPath((a)-[:K*]-(b)) RETURN p IS NULL AS n",
          "MATCH (a:P {id:1}) OPTIONAL MATCH (a)-[:K]->(c) RETURN c.id AS c"]:
    try: print('OK ',t,[list(r.values()) for r in db.execute(t)])
    except Exception as e: print('ERR',t,'->',str(e)[:140])
db.close()
```

### Actual

```text
OK  MATCH (a:P {id:1}),(b:P {id:2}) MATCH p=shortestPath((a)-[:K*]-(b)) RETURN length(p) AS l [[1]]
ERR MATCH (a:P {id:1}),(b:P {id:2}) OPTIONAL MATCH p=shortestPath((a)-[:K*]-(b)) RETURN length(p) AS l -> runtime error: subquery body contains an operation not yet supported inside CALL/EXISTS/COUNT subqueries: shortest path
ERR MATCH (a:P {id:1}),(b:P {id:3}) OPTIONAL MATCH p=shortestPath((a)-[:K*]-(b)) RETURN p IS NULL AS n -> runtime error: subquery body contains an operation not yet supported inside CALL/EXISTS/COUNT subqueries: shortest path
OK  MATCH (a:P {id:1}) OPTIONAL MATCH (a)-[:K]->(c) RETURN c.id AS c [[2]]
```

### Expected

`[[1]]` for the second query and `[[True]]` for the third: a path when there is one, and null when there is none. This
is the shape of LDBC SNB Interactive Complex 13.

### Found by

graph-bench's `snb_ic13`, <results page link>.

---

## Not filed (by design or performance only)

- Lookups without a label, such as `MATCH (m {id: $id})`, have no index to use, since `create_index` takes a label:
  performance only.
- `gds.pageRank` returns GDS's unnormalized scale: documented in `docs/reference/apoc-gds-compat.md`.
