# Docker Setup for Graph-Bench

## Quick Start

```bash
# Start all server databases (fresh containers, no persistence)
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Stop all
docker compose down
```

## Database Overview

| Database | Type | Query Language | Port | Notes |
|----------|------|----------------|------|-------|
| Neo4j | Server (Docker) | Cypher | 7687 | Enterprise + GDS algorithms |
| Memgraph | Server (Docker) | Cypher | 7688 | With MAGE algorithms |
| ArangoDB | Server (Docker) | AQL | 8529 | Multi-model |
| FalkorDB | Server (Docker) | Cypher | 6379 | Redis-based graph DB |
| NebulaGraph | Server (Docker) | nGQL | 9669 | Distributed graph DB |
| TuGraph | Server (Docker) | Cypher | 7689 | 34+ built-in algorithms |
| LadybugDB | Embedded | Cypher | N/A | `pip install real_ladybug` |
| Grafeo | Embedded | GQL (ISO) | N/A | `pip install grafeo` |

## Server Databases (Docker)

| Service | Image | Ports | Web UI |
|---------|-------|-------|--------|
| Neo4j | `neo4j:5-enterprise` | 7687 (bolt), 7474 (http) | http://localhost:7474 |
| Memgraph | `memgraph/memgraph-mage` | 7688 (bolt), 3000 (lab) | http://localhost:3000 |
| ArangoDB | `arangodb:latest` | 8529 | http://localhost:8529 |
| FalkorDB | `falkordb/falkordb` | 6379 | N/A |
| NebulaGraph | `vesoft/nebula-*` | 9669 | N/A |
| TuGraph | `tugraph/tugraph-runtime-centos7` | 7689 (bolt), 7070 (http), 9090 (rpc) | http://localhost:7070 |

### Credentials

| Database | User | Password |
|----------|------|----------|
| Neo4j | `neo4j` | `benchmark` |
| Memgraph | (none) | (none) |
| ArangoDB | `root` | `benchmark` |
| NebulaGraph | `root` | `nebula` |
| TuGraph | `admin` | `73@TuGraph` |

## Embedded Databases (No Docker)

These run directly in Python - no server needed:

```bash
# Install embedded adapters
pip install ladybug grafeo
```

- **ladybug**: Cypher queries, stored in `./data/ladybug`
- **Grafeo**: Cypher-like queries, stored in `./data/grafeo`

## Fresh Containers

Docker containers start fresh each time (no persistent volumes). This ensures:
- Consistent benchmark conditions
- No stale data between runs
- The benchmark suite clears data before each run anyway

## Feature Availability

| Feature | Neo4j+GDS | Memgraph+MAGE | ArangoDB | LadybugDB | Grafeo | TuGraph |
|---------|-----------|---------------|----------|-----------|--------|---------|
| Basic CRUD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Traversal (BFS/DFS) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Shortest Path | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PageRank | ✅ | ✅ | ❌ | ❌* | ✅ | ✅ |
| Community Detection | ✅ | ✅ | ❌ | ❌* | ✅ | ✅ |
| WCC | ✅ | ✅ | ❌ | ❌* | ✅ | ✅ |
| LCC | ✅ | ✅ | ❌ | ❌* | ✅ | ✅ |
| SSSP | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Legend:** ✅ Native | ❌ Not available | ❌* Uses NetworkX fallback

## Neo4j GDS Library

Neo4j Enterprise includes the Graph Data Science (GDS) library with algorithms:
- `gds.pageRank` - PageRank centrality
- `gds.louvain` - Community detection (Louvain)
- `gds.labelPropagation` - Label Propagation
- `gds.wcc` - Weakly Connected Components
- `gds.localClusteringCoefficient` - Local Clustering Coefficient
- `gds.shortestPath.dijkstra` - Single-Source Shortest Path

## Memgraph MAGE

Memgraph MAGE includes graph algorithms:
- `pagerank.get()` - PageRank
- `community_detection.get()` - Louvain community detection
- `label_propagation.get()` - Label Propagation
- `weakly_connected_components.get()` - WCC
- `clustering_coefficient.get()` - Clustering coefficients

## Existing Neo4j Container

If you already have Neo4j running in Docker Desktop:

1. Comment out the `neo4j` service in `docker-compose.yml`
2. Update `.env` with your existing Neo4j credentials
3. Make sure port 7687 is exposed

## Troubleshooting

### Port conflicts
```bash
# Windows
netstat -ano | findstr :7687

# Linux/Mac
lsof -i :7687
```

### Container won't start
```bash
docker compose logs neo4j
docker compose logs memgraph
docker compose logs arangodb
```

### Reset containers
```bash
docker compose down
docker compose up -d
```
