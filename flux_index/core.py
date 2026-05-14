"""
flux-index core — self-contained semantic code search engine.

Zero external dependencies. No model downloads. No GPU. No API calls.

Architecture:
  Source → Extract tiles (functions, classes, commits, files)
  Tiles → Embed (weighted word + n-gram hashing → fixed-dim vector)
  Query → Embed → Cosine similarity → Top-K results

The key insight: weight words 10× over character n-grams, and give
extra weight to identifiers (function names, class names, docstrings).
"""

from __future__ import annotations
import json, math, os, re, subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


# ─── Data Types ───────────────────────────────────────────────────

@dataclass
class Tile:
    """A single searchable unit extracted from source code."""
    id: str
    type: str          # function, class, struct, commit, file, readme
    path: str
    name: str
    content: str
    language: str = ""
    line: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A search hit."""
    tile: Tile
    score: float


# ─── Embedding Engine ────────────────────────────────────────────

class Embedder:
    """
    Word-level sparse embedding with IDF weights.
    
    Key insight: Instead of hashing words into a fixed-dim vector
    (which loses information to collisions), we use a sparse approach:
    each unique word gets its own dimension, weighted by IDF.
    
    For cosine similarity, sparse vectors work great — two documents
    sharing rare words get high similarity even without hashing.
    
    Feature channels:
      - Identifiers (function/class names): 10× weight
      - Words from content/docstrings: 3× weight  
      - Character bigrams: 1× weight (handles partial matches)
    """
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.idf: Dict[str, float] = {}
        self.vocab: Dict[str, int] = {}  # feature → dimension index
        self._next_dim = 0
    
    def _alloc_dim(self, feature: str) -> int:
        """Allocate a dimension for a feature."""
        if feature not in self.vocab:
            # Wrap around if we exceed max dim (but 128 is usually enough for vocab)
            self.vocab[feature] = self._next_dim % self.dim
            self._next_dim += 1
        return self.vocab[feature]
    
    def train(self, documents: List[str]):
        """Learn IDF weights and build vocabulary from corpus."""
        n = len(documents)
        freq = Counter()
        
        for doc in documents:
            features = set(self._feature_keys(doc))
            for f in features:
                freq[f] += 1
                self._alloc_dim(f)
        
        self.idf = {f: math.log(n / (df + 1)) for f, df in freq.items()}
    
    def embed(self, text: str) -> List[float]:
        """Convert text to a fixed-dim embedding vector."""
        vec = [0.0] * self.dim
        
        for feat, weight in self._weighted_features(text):
            idx = self._alloc_dim(feat)
            idf_w = self.idf.get(feat, 1.0)
            vec[idx] += weight * idf_w
        
        mag = math.sqrt(sum(x * x for x in vec))
        return [x / mag for x in vec] if mag > 0 else vec
    
    def _weighted_features(self, text: str) -> List[Tuple[str, float]]:
        """Extract features with importance weights."""
        features = []
        lower = text.lower()
        lines = lower.split("\n")
        
        # 1. Identifiers from first line (name), weight 15×
        first_line = lines[0] if lines else lower
        # Split CamelCase and snake_case
        idents = re.findall(r'[a-z]+', re.sub(r'([A-Z])', r' \1', first_line))
        for ident in idents:
            if len(ident) > 2:
                features.append((f"id:{ident}", 15.0))
        
        # 2. Words from all content, weight 5×
        words = re.findall(r'[a-z_]{2,}', lower)
        for w in words:
            if len(w) > 2:
                features.append((f"w:{w}", 5.0))
        
        # 3. Character bigrams for fuzzy matching, weight 1×
        padded = f"^{lower}$"
        for i in range(len(padded) - 1):
            features.append((f"b:{padded[i:i+2]}", 1.0))
        
        return features
    
    def _feature_keys(self, text: str) -> List[str]:
        """All unique feature keys (for IDF counting)."""
        return [f for f, _ in self._weighted_features(text)]


# ─── Vector Index ─────────────────────────────────────────────────

class Index:
    """
    In-memory vector index for fast cosine search.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.embedder = Embedder(dim)
        self.tiles: List[Tile] = []
        self.vectors: List[List[float]] = []
        self._norms: List[float] = []
    
    def add(self, tiles: List[Tile]):
        """Add tiles to the index. Trains embedder if first batch."""
        if not self.tiles:
            # Train on tile content
            docs = [f"{t.name} {t.content}" for t in tiles]
            self.embedder.train(docs)
        
        for t in tiles:
            text = f"{t.name}\n{t.content}"
            vec = self.embedder.embed(text)
            self.tiles.append(t)
            self.vectors.append(vec)
            self._norms.append(math.sqrt(sum(x * x for x in vec)))
    
    def search(self, query: str, top_k: int = 10, min_score: float = 0.0, prefer_tests: bool = False) -> List[SearchResult]:
        """Semantic search: embed query → cosine similarity → top-K."""
        q = self.embedder.embed(query)
        q_norm = math.sqrt(sum(x * x for x in q))
        if q_norm == 0:
            return []
        
        scored: List[Tuple[int, float]] = []
        for i, v in enumerate(self.vectors):
            if self._norms[i] == 0:
                continue
            dot = sum(q[j] * v[j] for j in range(self.dim))
            score = dot / (q_norm * self._norms[i])
            
            # Boost non-test tiles (tests are less useful as search results)
            if not prefer_tests:
                t = self.tiles[i]
                is_test = "test" in t.path.lower() or "test" in t.name.lower()
                if is_test:
                    score *= 0.7  # Dampen test results
            
            if score >= min_score:
                scored.append((i, score))
        
        scored.sort(key=lambda x: -x[1])
        return [SearchResult(tile=self.tiles[i], score=s) for i, s in scored[:top_k]]
    
    @property
    def count(self) -> int:
        return len(self.tiles)
    
    def save(self, path: str):
        """Save index to .fvt file."""
        data = {
            "version": "0.2.0",
            "dim": self.dim,
            "idf": {k: v for k, v in self.embedder.idf.items()},
            "vocab": {k: v for k, v in self.embedder.vocab.items()},
            "next_dim": self.embedder._next_dim,
            "tiles": [
                {
                    "id": t.id, "type": t.type, "path": t.path,
                    "name": t.name, "content": t.content[:2000],
                    "language": t.language, "line": t.line,
                    "metadata": t.metadata,
                }
                for t in self.tiles
            ],
            "vectors": self.vectors,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: str) -> int:
        """Load index from .fvt file. Returns tile count."""
        with open(path) as f:
            data = json.load(f)
        
        self.dim = data.get("dim", 128)
        self.embedder = Embedder(self.dim)
        self.embedder.idf = data.get("idf", {})
        self.embedder.vocab = data.get("vocab", {})
        self.embedder._next_dim = data.get("next_dim", len(self.embedder.vocab))
        self.tiles = []
        self.vectors = data.get("vectors", [])
        self._norms = []
        
        for t_data in data.get("tiles", []):
            self.tiles.append(Tile(
                id=t_data["id"], type=t_data["type"], path=t_data["path"],
                name=t_data["name"], content=t_data.get("content", ""),
                language=t_data.get("language", ""), line=t_data.get("line", 0),
                metadata=t_data.get("metadata", {}),
            ))
        
        for v in self.vectors:
            self._norms.append(math.sqrt(sum(x * x for x in v)))
        
        return len(self.tiles)


# ─── Source Extractors ────────────────────────────────────────────

def extract_py(filepath: str, content: str) -> List[Tile]:
    """Extract functions and classes from Python source."""
    tiles = []
    lines = content.split("\n")
    
    for match in re.finditer(r'^(class |async def |def )(\w+)', content, re.MULTILINE):
        name = match.group(2)
        kind = "class" if match.group(1) == "class " else "function"
        start = content[:match.start()].count("\n")
        
        # Extract docstring (highest semantic value)
        docstring = ""
        body_start = start + 1
        if body_start < len(lines):
            stripped = lines[body_start].strip()
            if stripped.startswith(('"""', "'''")):
                quote = stripped[:3]
                # Single-line docstring
                if stripped.count(quote) >= 2 and len(stripped) > 6:
                    docstring = stripped[3:-3].strip()
                else:
                    # Multi-line docstring
                    doc_lines = [stripped[3:]]
                    for i in range(body_start + 1, min(body_start + 30, len(lines))):
                        if lines[i].strip().endswith(quote):
                            doc_lines.append(lines[i].strip()[:-3])
                            break
                        doc_lines.append(lines[i])
                    docstring = " ".join(l.strip() for l in doc_lines).strip()
        
        # Get function body (for content)
        end = start + 1
        indent = len(lines[start]) - len(lines[start].lstrip())
        for i in range(start + 1, min(start + 60, len(lines))):
            if lines[i].strip() and not lines[i].strip().startswith(('"""', "'''", "#")):
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                if curr_indent <= indent and re.match(r'^(class |async def |def |@)', lines[i].lstrip()):
                    break
            end = i + 1
        
        # Build rich content: name + docstring + signature
        # Docstring is the most important semantic content
        sig_line = lines[start].strip()
        if docstring:
            body = f"{sig_line}\n{docstring}"
        else:
            body = "\n".join(lines[start:end])
        
        tiles.append(Tile(
            id=f"py:{filepath}:{name}",
            type=kind, path=filepath, name=name,
            content=body[:2000], language=".py", line=start + 1,
        ))
    
    if not tiles:
        tiles.append(Tile(
            id=f"py:{filepath}:file",
            type="file", path=filepath, name=Path(filepath).name,
            content=content[:2000], language=".py",
        ))
    
    return tiles


def extract_rs(filepath: str, content: str) -> List[Tile]:
    """Extract Rust fns, structs, enums, impls."""
    tiles = []
    lines = content.split("\n")
    
    for match in re.finditer(
        r'^\s*(pub\s+)?(async\s+)?fn\s+(\w+)|^\s*(pub\s+)?struct\s+(\w+)|^\s*(pub\s+)?enum\s+(\w+)',
        content, re.MULTILINE
    ):
        name = match.group(3) or match.group(5) or match.group(7)
        if not name:
            continue
        kind = "function" if match.group(3) else "struct"
        start = content[:match.start()].count("\n")
        end = min(start + 40, len(lines))
        body = "\n".join(lines[start:end])
        
        tiles.append(Tile(
            id=f"rs:{filepath}:{name}",
            type=kind, path=filepath, name=name,
            content=body[:2000], language=".rs", line=start + 1,
        ))
    
    if not tiles:
        tiles.append(Tile(
            id=f"rs:{filepath}:file",
            type="file", path=filepath, name=Path(filepath).name,
            content=content[:2000], language=".rs",
        ))
    
    return tiles


def extract_c(filepath: str, content: str) -> List[Tile]:
    """Extract C functions and struct definitions."""
    tiles = []
    lines = content.split("\n")
    
    for match in re.finditer(r'^[\w\s\*]+?\s+(\w+)\s*\([^)]*\)\s*\{', content, re.MULTILINE):
        name = match.group(1)
        if name in ("if", "while", "for", "switch", "return", "sizeof"):
            continue
        start = content[:match.start()].count("\n")
        end = min(start + 40, len(lines))
        body = "\n".join(lines[start:end])
        
        tiles.append(Tile(
            id=f"c:{filepath}:{name}",
            type="function", path=filepath, name=name,
            content=body[:2000], language=".c", line=start + 1,
        ))
    
    if not tiles:
        tiles.append(Tile(
            id=f"c:{filepath}:file",
            type="file", path=filepath, name=Path(filepath).name,
            content=content[:2000], language=".c",
        ))
    
    return tiles


def extract_js(filepath: str, content: str) -> List[Tile]:
    """Extract JS/TS functions and classes."""
    tiles = []
    lines = content.split("\n")
    
    for match in re.finditer(
        r'(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)|(?:export\s+)?class\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z])\s*=>',
        content, re.MULTILINE
    ):
        name = match.group(1) or match.group(2) or match.group(3)
        if not name:
            continue
        kind = "class" if match.group(2) else "function"
        start = content[:match.start()].count("\n")
        end = min(start + 30, len(lines))
        body = "\n".join(lines[start:end])
        
        tiles.append(Tile(
            id=f"js:{filepath}:{name}",
            type=kind, path=filepath, name=name,
            content=body[:2000], language=Path(filepath).suffix, line=start + 1,
        ))
    
    if not tiles:
        tiles.append(Tile(
            id=f"js:{filepath}:file",
            type="file", path=filepath, name=Path(filepath).name,
            content=content[:2000], language=Path(filepath).suffix,
        ))
    
    return tiles


# ─── Extractor Dispatch ──────────────────────────────────────────

EXTRACTORS = {
    ".py": extract_py,
    ".rs": extract_rs,
    ".c": extract_c, ".h": extract_c, ".cpp": extract_c, ".hpp": extract_c,
    ".js": extract_js, ".ts": extract_js, ".tsx": extract_js,
}

SKIP_DIRS = {".git", "node_modules", "__pycache__", "target", "build", "dist", ".venv", "venv", ".flux"}
SKIP_PATTERNS = {"__init__.py", "setup.py", "conftest.py", "pyproject.toml", "setup.cfg"}


def extract_repo(path: str, max_commits: int = 200) -> List[Tile]:
    """Extract all searchable tiles from a repo."""
    tiles = []
    
    # README
    for name in ["README.md", "README.rst", "README.txt", "README"]:
        rpath = os.path.join(path, name)
        if os.path.exists(rpath):
            with open(rpath, errors="ignore") as f:
                tiles.append(Tile(
                    id=f"readme:README", type="readme", path=name,
                    name="README", content=f.read()[:5000],
                ))
            break
    
    # Source files
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in sorted(files):
            if fname in SKIP_PATTERNS:
                continue
            ext = Path(fname).suffix
            if ext in EXTRACTORS:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, path)
                try:
                    with open(fpath, errors="ignore") as f:
                        content = f.read()
                    tiles.extend(EXTRACTORS[ext](rel, content))
                except:
                    pass
    
    # Commits
    try:
        result = subprocess.run(
            ["git", "log", f"-{max_commits}", "--format=%H|%s"],
            capture_output=True, text=True, cwd=path, timeout=30,
        )
        for line in result.stdout.strip().split("\n"):
            if "|" not in line:
                continue
            sha, msg = line.split("|", 1)
            tiles.append(Tile(
                id=f"commit:{sha[:10]}", type="commit",
                path=f"commit:{sha[:10]}", name=msg[:100],
                content=msg,
            ))
    except:
        pass
    
    return tiles


def index_repo(repo_path: str, output: str = None, dim: int = 64, max_commits: int = 200) -> dict:
    """
    Index a repo into a searchable .fvt file.
    
    Returns stats dict.
    """
    repo_name = Path(repo_path).resolve().name
    if output is None:
        output = os.path.join(repo_path, ".flux.fvt")
    
    # Extract
    tiles = extract_repo(repo_path, max_commits)
    
    # Index
    idx = Index(dim)
    idx.add(tiles)
    idx.save(output)
    
    types = Counter(t.type for t in tiles)
    langs = Counter(t.language for t in tiles if t.language)
    
    return {
        "repo": repo_name,
        "tiles": len(tiles),
        "types": dict(types),
        "languages": dict(langs),
        "output": output,
        "size_kb": os.path.getsize(output) / 1024,
    }
