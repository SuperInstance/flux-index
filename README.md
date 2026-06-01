# flux-index

**Semantic code search with zero dependencies.** Index any repository into a searchable vector space, query with natural language, get ranked results — no model downloads, no GPU, no API calls.

## What This Does

Flux-index turns a code repository into a searchable "vector twin" (`.fvt` file). It walks the repo, extracts every meaningful unit — functions, classes, structs, commits, the README — embeds each one as a sparse TF-IDF-weighted vector, and saves the result as a single file. At query time, it embeds your query with the same vocabulary and runs cosine similarity against the index. Pure Python, no external services.

The library gives you:

- **Repository indexing** — extract tiles (functions, classes, files, commits, README) from Python, Rust, C/C++, and JS/TS repos
- **Semantic search** — TF-IDF-weighted word + character n-gram embeddings with identifier boosting
- **CRDT sync** — delta-state OR-Set for conflict-free multi-machine index synchronisation
- **CLI** — `flux-index` command for index/search/map operations
- **Zero dependencies** — no database, no model downloads, no GPU, no API calls

## Key Idea

Instead of using a neural embedding model (which requires GPU, downloads, and API latency), flux-index builds a **sparse embedding** from three feature channels with different importance weights:

| Channel | Weight | What it captures |
|---|---|---|
| Identifiers (function/class names) | 15× | "What is this called?" — highest signal |
| Words from content/docstrings | 5× | "What does it do?" — semantic content |
| Character bigrams | 1× | "What does it look like?" — fuzzy matching |

IDF weights trained on the corpus make rare words (discriminative terms) count more than common ones. The result is a 128-dimensional embedding that captures enough semantic signal for code search at ~0.1ms query latency.

## Install

```bash
pip install flux-index
```

Requires Python ≥ 3.8. No external dependencies.

## Quick Start

### CLI

```bash
# Index a repository
flux-index /path/to/my-repo

# Search it
flux-index search "authentication flow"

# Search all indexed repos
flux-index search --all "parse config"

# Show codebase map
flux-index map

# Find similar code
flux-index similar "database connection"
```

### Python API

```python
from flux_index import index_repo, Index

# Index a repo → saves .flux.fvt file
stats = index_repo("/path/to/my-repo")
print(f"Indexed {stats['tiles']} tiles")
# Files: 42, functions: 187, classes: 23, commits: 200

# Load and search
idx = Index()
idx.load("/path/to/my-repo/.flux.fvt")

results = idx.search("error handling middleware", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.tile.type}: {r.tile.name} ({r.tile.path}:{r.tile.line})")
    # [0.847] function: handle_error (src/middleware.py:42)
```

### CRDT Sync (multi-machine)

```python
from flux_index.crdt import CRDTIndex
from flux_index.core import Tile

# Machine A: index and export
crdt_a = CRDTIndex(replica_id="machine-a")
delta = crdt_a.add_tiles([Tile(id="t1", type="function", ...)])

# Machine B: merge changes
crdt_b = CRDTIndex(replica_id="machine-b")
changes = crdt_b.merge(delta)  # Idempotent — merge twice = no-op
```

## API Reference

### `flux_index.core` — Indexing & Search

| Class / Function | Description |
|---|---|
| `Tile` | A single searchable unit (function, class, file, commit, README) |
| `SearchResult` | A search hit with `.tile` and `.score` |
| `Index(dim=64)` | In-memory vector index |
| `Index.add(tiles)` | Add tiles (trains embedder on first batch) |
| `Index.search(query, top_k=10, min_score=0.0)` | Semantic search |
| `Index.save(path)` / `Index.load(path)` | Persist to/from `.fvt` JSON |
| `extract_repo(path, max_commits=200)` | Extract tiles from a repository |
| `index_repo(repo_path, output, dim, max_commits)` | Full pipeline: extract → embed → save |

### `flux_index.search` — FluxVectorTwin

| Class / Function | Description |
|---|---|
| `FluxVectorTwin(dim=128)` | Tiny embedding store for tile search |
| `FluxVectorTwin.train(texts)` | Compute IDF weights from corpus |
| `FluxVectorTwin.index_tiles(tiles)` | Index a batch of tiles |
| `FluxVectorTwin.search(query, top_k)` | Cosine similarity search |
| `FluxVectorTwin.search_room(query, room)` | Room-scoped search |
| `FluxVectorTwin.similar_to(tile_id)` | Find tiles similar to a given one |
| `text_to_embedding(text, idf_weights, dim)` | Raw text → embedding vector |

### `flux_index.crdt` — Conflict-Free Sync

| Class / Function | Description |
|---|---|
| `Dot(replica, seq)` | Causal marker for CRDT operations |
| `Delta` | Incremental state change (added/removed tiles, relevance updates) |
| `CRDTIndex(replica_id, dim=128)` | CRDT-wrapped index |
| `CRDTIndex.add_tiles(tiles)` | Add tiles locally, returns Delta for sync |
| `CRDTIndex.merge(delta)` | Merge a remote delta (idempotent) |
| `CRDTIndex.increment_relevance(tile_id)` | G-Counter increment for popularity tracking |
| `CRDTIndex.search(query, top_k)` | Search with relevance boost |

### `flux_index.extractor` — Repo to Vector Space

| Function | Description |
|---|---|
| `repo_to_vectors(repo_url_or_path, ...)` | Clone (if URL) → extract → embed → save `.fvt` |
| `search_repo(fvt_path, query)` | Search a pre-indexed `.fvt` file |
| `repo_report(fvt_path)` | Human-readable summary |

### `flux_index.cli` — Command Line

```
flux-index <path>                    Index a repository
flux-index search "query"            Search indexed repo
flux-index search --all "query"      Search all indexed repos
flux-index map                       Show codebase map
flux-index similar "reference"       Find similar code
```

### Language Extractors

| Language | What's extracted |
|---|---|
| Python (`.py`) | Functions (including async), classes, docstrings |
| Rust (`.rs`) | `fn`, `pub fn`, `async fn`, `struct`, `enum` |
| C/C++ (`.c`, `.h`, `.cpp`, `.hpp`) | Function definitions, struct-like patterns |
| JS/TS (`.js`, `.ts`, `.tsx`) | Functions, classes, arrow functions, exports |
| All others | Whole-file tiles |

## How It Works

```
┌──────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────┐
│  Source   │────▶│   Extractor  │────▶│   Embedder    │────▶│  Index   │
│  repo     │     │              │     │               │     │  (.fvt)  │
│           │     │ extract_py() │     │ 3 channels:   │     │          │
│ .py .rs   │     │ extract_rs() │     │ id: 15×       │     │ tiles[]  │
│ .c  .js   │     │ extract_c()  │     │ words: 5×     │     │ vecs[]   │
│ README    │     │ extract_js() │     │ bigrams: 1×   │     │ IDF{}    │
│ git log   │     │              │     │               │     │ vocab{}  │
└──────────┘     └──────────────┘     └───────┬───────┘     └────┬─────┘
                                              │                   │
                                              ▼                   ▼
                                     ┌───────────────────────────────┐
                                     │         Query Time            │
                                     │                               │
                                     │  query → embed → cosine sim   │
                                     │  → top-K → SearchResult[]     │
                                     │                               │
                                     │  Latency: ~0.1ms              │
                                     └───────────────────────────────┘
```

### Embedding pipeline

1. **Extract**: Walk the repo, parse each source file with a language-specific extractor, pull out functions/classes/structs with their names, signatures, and docstrings. Also extract git commit messages and the README.

2. **Train**: On first batch, compute IDF (Inverse Document Frequency) weights across all extracted tiles. Each unique feature (word, identifier, bigram) gets a weight proportional to `log(N / df)` — rare features get high weight.

3. **Embed**: For each tile, project its name + content into a fixed-dimension vector using three weighted feature channels. The identifier channel (15×) ensures that searching for "authenticate" strongly matches functions named `authenticate_user`.

4. **Search**: Embed the query the same way, compute cosine similarity against all stored vectors, return top-K.

### CRDT layer

The CRDT layer wraps the index in an Observed-Remove Set (OR-Set) with:
- **Dot-based causal markers** — each write gets a unique (replica_id, sequence_number) marker
- **Add-wins semantics** — concurrent add and remove → add wins
- **Semantic dedup** — new tiles with >95% cosine similarity to existing ones are treated as updates (last-writer-wins)
- **G-Counter relevance tracking** — search-hit counts propagate across machines, boosting popular tiles

## The Math

### Sparse TF-IDF embedding

For a document (tile) with text T:

```
v[i] = Σ (weight(feature) × idf(feature))  for each feature → dim i
```

Where features are:
- `id:word` from identifiers (weight = 15)
- `w:word` from content (weight = 5)
- `b:ch` character bigrams (weight = 1)

IDF for feature f across N documents:

```
idf(f) = log(N / (df(f) + 1))
```

The vector is L2-normalised so cosine similarity is just a dot product.

### Cosine similarity

```
sim(q, d) = (q · d) / (‖q‖ × ‖d‖)
```

Pre-computed norms make this a single dot product + two lookups at query time.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

84 tests across 4 test files covering core indexing, search, CRDT sync, and extractors.

## License

MIT
