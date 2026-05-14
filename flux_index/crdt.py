"""
flux-index CRDT layer — delta-state OR-Set for index synchronization.

Multi-machine index sync without conflicts. No central server.

CRDT primitives:
  - Dot-based OR-Set (add-wins, observed-remove)
  - LWW-Register per tile (last-writer-wins)
  - G-Counter per tile (relevance tracking)
"""

from __future__ import annotations
import json, math, os, socket, uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from flux_index.core import Tile, SearchResult, Embedder


@dataclass(frozen=True)
class Dot:
    """Causal marker: {replica_id, sequence_number}."""
    replica: str
    seq: int
    
    def to_dict(self) -> dict:
        return {"r": self.replica, "s": self.seq}
    
    @staticmethod
    def from_dict(d: dict) -> "Dot":
        return Dot(d["r"], d["s"])


@dataclass
class Delta:
    """Incremental state change to an index."""
    dots: Set[Dot] = field(default_factory=set)
    added: List[dict] = field(default_factory=list)    # [{dot, tile, vector}]
    removed: Set[Dot] = field(default_factory=set)
    relevance: Dict[str, int] = field(default_factory=dict)


class CRDTIndex:
    """
    CRDT-wrapped index for multi-machine sync.
    
    Stores tiles as a dict (not parallel arrays) for safe mutation.
    Uses an Embedder for query embedding and semantic dedup.
    """
    
    def __init__(self, replica_id: str = None, dim: int = 128):
        self.replica_id = replica_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.dim = dim
        self.embedder = Embedder(dim)
        self._seq: int = 0
        self._tiles: Dict[str, dict] = {}  # tile_id → {tile, vector, dot, norm}
        self._seen_dots: Set[Dot] = set()
        self._relevance: Dict[str, int] = {}  # G-Counter (max semantics)
    
    @property
    def count(self) -> int:
        return len(self._tiles)
    
    def next_dot(self) -> Dot:
        self._seq += 1
        return Dot(self.replica_id, self._seq)
    
    # ─── Local operations ────────────────────────────────────────
    
    def add_tiles(self, tiles: List[Tile]) -> Delta:
        """
        Add tiles locally. Trains embedder if first batch.
        Returns a Delta for sync.
        """
        if not self._tiles:
            self.embedder.train([f"{t.name}\n{t.content}" for t in tiles])
        
        delta = Delta()
        for tile in tiles:
            vector = self.embedder.embed(f"{tile.name}\n{tile.content}")
            norm = math.sqrt(sum(x * x for x in vector))
            dot = self.next_dot()
            
            self._tiles[tile.id] = {
                "tile": tile,
                "vector": vector,
                "dot": dot,
                "norm": norm,
            }
            self._seen_dots.add(dot)
            
            delta.dots.add(dot)
            delta.added.append({
                "dot": dot.to_dict(),
                "tile": {
                    "id": tile.id, "type": tile.type, "path": tile.path,
                    "name": tile.name, "content": tile.content[:2000],
                    "language": tile.language, "line": tile.line,
                },
                "vector": vector,
            })
        
        return delta
    
    def increment_relevance(self, tile_id: str, count: int = 1) -> Delta:
        """G-Counter increment. Returns a delta."""
        self._relevance[tile_id] = self._relevance.get(tile_id, 0) + count
        delta = Delta()
        delta.relevance[tile_id] = count
        return delta
    
    # ─── Merge ───────────────────────────────────────────────────
    
    def merge(self, delta: Delta) -> int:
        """
        Merge a remote delta. Returns number of changes applied.
        Idempotent: merging same delta twice = no-op.
        """
        changes = 0
        
        # Apply additions
        for entry in delta.added:
            dot = Dot.from_dict(entry["dot"])
            if dot in self._seen_dots:
                continue  # Idempotent
            
            self._seen_dots.add(dot)
            tile_data = entry["tile"]
            tile = Tile(
                id=tile_data["id"], type=tile_data["type"], path=tile_data["path"],
                name=tile_data["name"], content=tile_data.get("content", ""),
                language=tile_data.get("language", ""), line=tile_data.get("line", 0),
            )
            vector = entry["vector"]
            norm = math.sqrt(sum(x * x for x in vector))
            
            # Semantic dedup: check if similar tile already exists
            existing_id = self._find_semantic_match(tile, vector)
            if existing_id and existing_id in self._tiles:
                # LWW: higher seq wins
                existing_dot = self._tiles[existing_id]["dot"]
                if dot.seq > existing_dot.seq:
                    self._tiles[tile.id] = {"tile": tile, "vector": vector, "dot": dot, "norm": norm}
                    changes += 1
                # else: keep existing, no change
            else:
                self._tiles[tile.id] = {"tile": tile, "vector": vector, "dot": dot, "norm": norm}
                changes += 1
        
        # Apply removals
        for dot in delta.removed:
            if dot not in self._seen_dots:
                continue
            to_remove = [tid for tid, entry in self._tiles.items() if entry["dot"] == dot]
            for tid in to_remove:
                del self._tiles[tid]
                changes += 1
        
        # Apply relevance (G-Counter: max semantics)
        for tile_id, count in delta.relevance.items():
            current = self._relevance.get(tile_id, 0)
            new_val = max(current, count)
            if new_val != current:
                self._relevance[tile_id] = new_val
                changes += 1
        
        return changes
    
    # ─── Search ──────────────────────────────────────────────────
    
    def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> List[SearchResult]:
        """Search with relevance boost."""
        q_vec = self.embedder.embed(query)
        q_norm = math.sqrt(sum(x * x for x in q_vec))
        if q_norm == 0:
            return []
        
        scored = []
        for tid, entry in self._tiles.items():
            vec = entry["vector"]
            norm = entry["norm"]
            if norm == 0:
                continue
            dot = sum(q_vec[j] * vec[j] for j in range(self.dim))
            score = dot / (q_norm * norm)
            
            # Test dampening
            tile = entry["tile"]
            if "test" in tile.path.lower() or "test" in tile.name.lower():
                score *= 0.7
            
            # Relevance boost (logarithmic)
            relevance = self._relevance.get(tid, 0)
            score += 0.05 * math.log(1 + relevance)
            
            if score >= min_score:
                scored.append(SearchResult(tile=tile, score=score))
        
        scored.sort(key=lambda r: -r.score)
        return scored[:top_k]
    
    # ─── Semantic dedup ──────────────────────────────────────────
    
    def _find_semantic_match(self, tile: Tile, vector: List[float], threshold: float = 0.95) -> Optional[str]:
        """Find semantically similar existing tile. Returns tile_id or None."""
        # Fast path: exact content match
        for tid, entry in self._tiles.items():
            if entry["tile"].content == tile.content and entry["tile"].type == tile.type:
                return tid
        
        # Slow path: embedding similarity
        vec_norm = math.sqrt(sum(x * x for x in vector))
        if vec_norm == 0:
            return None
        
        best_score = 0.0
        best_id = None
        for tid, entry in self._tiles.items():
            if entry["tile"].type != tile.type:
                continue
            existing_vec = entry["vector"]
            existing_norm = entry["norm"]
            if existing_norm == 0:
                continue
            dot = sum(vector[j] * existing_vec[j] for j in range(self.dim))
            score = dot / (vec_norm * existing_norm)
            if score > best_score:
                best_score = score
                best_id = tid
        
        return best_id if best_score >= threshold else None
    
    # ─── Serialization ──────────────────────────────────────────
    
    def save(self, fvt_path: str):
        """Save base index + CRDT metadata."""
        # Save base index using Index.save()
        from flux_index.core import Index
        base = Index(self.dim)
        for entry in self._tiles.values():
            base.tiles.append(entry["tile"])
            base.vectors.append(entry["vector"])
            base._norms.append(entry["norm"])
        base.embedder.idf = self.embedder.idf
        base.embedder.vocab = self.embedder.vocab
        base.embedder._next_dim = self.embedder._next_dim
        base.save(fvt_path)
        
        # Save CRDT metadata
        crdt_data = {
            "replica": self.replica_id,
            "seq": self._seq,
            "tile_dots": {tid: entry["dot"].to_dict() for tid, entry in self._tiles.items()},
            "seen_dots": [d.to_dict() for d in self._seen_dots],
            "relevance": self._relevance,
        }
        crdt_path = fvt_path + ".crdt"
        with open(crdt_path, "w") as f:
            json.dump(crdt_data, f)
    
    def load(self, fvt_path: str):
        """Load base index + CRDT metadata."""
        from flux_index.core import Index
        base = Index(self.dim)
        count = base.load(fvt_path)
        
        # Rebuild internal state from base index
        self.embedder = base.embedder
        self._tiles = {}
        for tile, vec, norm in zip(base.tiles, base.vectors, base._norms):
            self._tiles[tile.id] = {
                "tile": tile, "vector": vec, "norm": norm,
                "dot": Dot("?", 0),  # Placeholder — overwritten if CRDT file exists
            }
        
        # Load CRDT metadata if available
        crdt_path = fvt_path + ".crdt"
        if os.path.exists(crdt_path):
            with open(crdt_path) as f:
                data = json.load(f)
            self.replica_id = data.get("replica", self.replica_id)
            self._seq = data.get("seq", 0)
            for tid, dot_dict in data.get("tile_dots", {}).items():
                if tid in self._tiles:
                    self._tiles[tid]["dot"] = Dot.from_dict(dot_dict)
            self._seen_dots = {Dot(d["r"], d["s"]) for d in data.get("seen_dots", [])}
            self._relevance = data.get("relevance", {})
