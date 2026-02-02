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
| Neo4j | Server (Docker) | Cypher | 7687 | Community Edition |
| Memgraph | Server (Docker) | Cypher | 7688 | With MAGE algorithms |
| ArangoDB | Server (Docker) | AQL | 8529 | Multi-model |
| Ladybugdb | Embedded | Cypher | N/A | `pip install real_ladybug` |
| DuckDB | Embedded | SQL | N/A | `pip install duckdb` |
| Grafeo | Embedded | Cypher-like | N/A | `pip install grafeo` |

## Server Databases (Docker)

| Service | Image | Ports | Web UI |
|---------|-------|-------|--------|
| Neo4j | `neo4j:5-community` | 7687 (bolt), 7474 (http) | http://localhost:7474 |
| Memgraph | `memgraph/memgraph-mage` | 7688 (bolt), 3000 (lab) | http://localhost:3000 |
| ArangoDB | `arangodb:latest` | 8529 | http://localhost:8529 |

### Credentials

| Database | User | Password |
|----------|------|----------|
| Neo4j | `neo4j` | `benchmark` |
| Memgraph | (none) | (none) |
| ArangoDB | `root` | `benchmark` |

## Embedded Databases (No Docker)

These run directly in Python - no server needed:

```bash
# Install embedded adapters
pip install ladybug duckdb grafeo
```

- **ladybug**: Cypher queries, stored in `./data/ladybug`
- **DuckDB**: SQL queries, in-memory by default (`:memory:`)
- **Grafeo**: Cypher-like queries, stored in `./data/grafeo`

## Fresh Containers

Docker containers start fresh each time (no persistent volumes). This ensures:
- Consistent benchmark conditions
- No stale data between runs
- The benchmark suite clears data before each run anyway

## Feature Availability (Community Editions)

| Feature | Neo4j | Memgraph | ArangoDB | ladybug | DuckDB | Grafeo |
|---------|-------|----------|----------|------|--------|--------|
| Basic CRUD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Traversal (BFS/DFS) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Shortest Path | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PageRank | ❌ | ✅ | ❌ | ❌ | ❌ | ⚡ |
| Community Detection | ❌ | ✅ | ❌ | ❌ | ❌ | ⚡ |

**Legend:** ✅ Available | ❌ Not available | ⚡ Depends on version

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
