# flux-index

Semantic code search engine with zero dependencies — index repositories as searchable tile collections, query with natural language, get ranked results.

## What This Gives You

- **Repository indexing** — walk a repo, extract tiles (functions, classes, modules) with metadata
- **Semantic search** — query by meaning, not just keywords
- **Ranked results** — results scored by relevance with snippet extraction
- **Zero dependencies** — pure Python, no database or external service required
- **Fast** — in-memory inverted index with O(1) lookups

## Quick Start

```python
from flux_index import index_repo, Index

# Index a repository
index = index_repo("/path/to/my-repo")
print(f"Indexed {len(index)} tiles")

# Search
results = index.search("constraint lattice snap")
for r in results:
    print(f"[{r.score:.3f}] {r.tile.file}:{r.tile.line} — {r.tile.name}")
```

### CLI

```bash
# Index a repo
python -m flux_index index /path/to/repo --output index.json

# Search
python -m flux_index search "lattice consonance" --index index.json
```

## API Reference

| Class / Function | Description |
|---|---|
| `Index` | In-memory search index |
| `Tile` | Indexed code unit (function, class, module) |
| `SearchResult` | Ranked search hit with score and snippet |
| `index_repo(path)` | Walk repo and build index |
| `extract_repo(path)` | Extract tiles from repository |

## How It Fits

The **code search layer** of the FLUX ecosystem:

- [eisenstein-embed](https://github.com/SuperInstance/eisenstein-embed) — Eisenstein embeddings for semantic matching
- [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) — lattice theory for distance metrics
- [flux-compiler-workspace](https://github.com/SuperInstance/flux-compiler-workspace) — searches compiled constraint programs

## Testing

```bash
pytest -v  # 4 test files
```

## Installation

```bash
pip install flux-index
```

Requires Python ≥ 3.10.

## License

MIT
