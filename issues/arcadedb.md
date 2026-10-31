# ArcadeDB: issues to file

Upstream: https://github.com/ArcadeData/arcadedb. The Python package used by graph-bench, `arcadedb-embedded`
(https://github.com/humemai/arcadedb-embedded-python), is only a wrapper around the ArcadeDB engine; the importer
is part of the engine, so the issue belongs to ArcadeData/arcadedb.
Latest stable on PyPI, checked 2026-10-31: `arcadedb-embedded` 26.10.2.post1.

One draft.

---

## 1. Graph Importer CSV reader does not honour quoted fields (commas and doubled quotes in values are mangled)

**Status:** draft. **Upstream:** no duplicate found (searched `ArcadeData/arcadedb`: "csv quotes", "importer quoted", "Graph Importer", "GraphImporter CSV quote", "quoted CSV", "RFC 4180"). Related but different: #7267 (closed; `CsvRowSource` split with a regex, so metacharacter delimiters break) and #8046 (closed; single-quoted fields in the default CSV importer). Link both in the issue so the maintainers see it is the same code path (`CsvRowSource`) with a new symptom.

### Environment

- `arcadedb-embedded` 26.10.2.post1 (PyPI, latest stable; ArcadeDB 26.10.2).
- Windows 11 Pro 10.0.26200, Python 3.12 (the same output was seen on Linux earlier).

### Minimal reproduction

```python
import json, pathlib, tempfile
import arcadedb_embedded as adb, jpype

adb.jvm.start_jvm()
d = pathlib.Path(tempfile.mkdtemp())
(d / "V.csv").write_text('id,name\n1,"Smith, John"\n2,"say ""hi"""\n3,plain\n', encoding="utf-8")
cfg = {"vertices": [{"type": "V", "file": "V.csv", "id": "id", "properties": {"id": "long:id", "name": "name"}}]}
GI = jpype.JClass("com.arcadedb.integration.importer.graph.GraphImporter")
J = jpype.JClass("com.arcadedb.serializer.json.JSONObject")(json.dumps(cfg))
db = adb.create_database(str(d / "db"))
GI.createSchemaFromConfig(db.get_java_database(), J)
GI.fromJSON(db.get_java_database(), J, str(d)).run()
print(db.query("opencypher", "MATCH (n:V) RETURN n.id AS id, n.name AS name ORDER BY id").to_list())
```

### Expected

RFC 4180: a field enclosed in double quotes may contain the delimiter, and `""` inside it stands for one quote. The
file holds the values `Smith, John` and `say "hi"`, so the query should print:
`[{'id': 1, 'name': 'Smith, John'}, {'id': 2, 'name': 'say "hi"'}, {'id': 3, 'name': 'plain'}]`.

The Graph Importer documentation
(https://docs.arcadedb.com/arcadedb/how-to/migration/graph-importer) does state that the CSV reader "splits on the
delimiter without honouring quotes" (under array columns). That makes it a documented limitation, but one that
silently corrupts ordinary exported CSV (names, titles and message text with commas or quotes) instead of failing, so
the request is either to honour quotes or to reject a row whose field count or quoting is wrong.

### Actual

```text
[{'id': 1, 'name': '"Smith'}, {'id': 2, 'name': '"say ""hi"""'}, {'id': 3, 'name': 'plain'}]
```

The first value is cut at the comma (`"Smith`, the rest `John"` is lost), the second keeps the quote characters
verbatim. No error or warning is raised (the import log reports "3 vertices" as complete).

### Found by

graph-bench (loading the LDBC SNB sample with the Graph Importer, the fastest ArcadeDB bulk loader, gave wrong text
values; the adapter loads with GraphBatch instead), <results page link>.

---

## Not filed (by design or performance only)

- Unlabeled id lookups scan every vertex (IS4, IS5, IS6): planner/performance only, answers agree.
- Algorithms slower than the 10x budget rule: performance only.
