"""Tests for flux-index CRDT layer."""

import os, tempfile, json
from flux_index.core import Tile
from flux_index.crdt import Dot, Delta, CRDTIndex


# ─── Dot ─────────────────────────────────────────────────────────

def test_dot_creation():
    d = Dot("replica-1", 5)
    assert d.replica == "replica-1"
    assert d.seq == 5

def test_dot_frozen():
    d = Dot("r", 1)
    try:
        d.replica = "x"
        assert False, "Should be frozen"
    except (AttributeError, TypeError):
        pass

def test_dot_to_dict_from_dict_roundtrip():
    d = Dot("host-42", 99)
    d2 = Dot.from_dict(d.to_dict())
    assert d2 == d

def test_dot_hashable():
    s = {Dot("a", 1), Dot("a", 1), Dot("b", 2)}
    assert len(s) == 2


# ─── Delta ────────────────────────────────────────────────────────

def test_delta_default():
    d = Delta()
    assert d.dots == set()
    assert d.added == []
    assert d.removed == set()
    assert d.relevance == {}


# ─── CRDTIndex basics ────────────────────────────────────────────

def test_crdt_index_empty():
    idx = CRDTIndex(replica_id="test-replica", dim=64)
    assert idx.count == 0
    assert idx.replica_id == "test-replica"

def test_crdt_index_add_tiles():
    idx = CRDTIndex(dim=64)
    tiles = [
        Tile(id="t1", type="function", path="a.py", name="foo", content="does foo things"),
        Tile(id="t2", type="function", path="b.py", name="bar", content="does bar things"),
    ]
    delta = idx.add_tiles(tiles)
    assert idx.count == 2
    assert len(delta.added) == 2
    assert len(delta.dots) == 2

def test_crdt_next_dot_monotonic():
    idx = CRDTIndex(replica_id="r", dim=64)
    d1 = idx.next_dot()
    d2 = idx.next_dot()
    assert d1.seq < d2.seq
    assert d1.replica == d2.replica == "r"


# ─── CRDT Search ─────────────────────────────────────────────────

def test_crdt_search():
    idx = CRDTIndex(dim=64)
    tiles = [
        Tile(id="auth", type="function", path="auth.py", name="login",
             content="handles user login authentication"),
        Tile(id="db", type="function", path="db.py", name="connect",
             content="database connection management"),
    ]
    idx.add_tiles(tiles)
    results = idx.search("user login", top_k=2)
    assert len(results) > 0
    assert results[0].tile.name == "login"

def test_crdt_search_empty():
    idx = CRDTIndex(dim=64)
    assert idx.search("anything", top_k=5) == []

def test_crdt_search_min_score():
    idx = CRDTIndex(dim=64)
    tiles = [
        Tile(id="t1", type="function", path="a.py", name="alpha",
             content="alpha beta gamma"),
    ]
    idx.add_tiles(tiles)
    results = idx.search("zzzzzzz totally unrelated", top_k=5, min_score=0.99)
    assert len(results) == 0


# ─── CRDT Relevance (G-Counter) ──────────────────────────────────

def test_increment_relevance():
    idx = CRDTIndex(dim=64)
    delta = idx.increment_relevance("tile-1", 3)
    assert delta.relevance == {"tile-1": 3}
    assert idx._relevance["tile-1"] == 3


# ─── CRDT Merge ──────────────────────────────────────────────────

def test_merge_adds_new_tiles():
    idx1 = CRDTIndex(replica_id="r1", dim=64)
    idx2 = CRDTIndex(replica_id="r2", dim=64)

    tiles1 = [Tile(id="a", type="function", path="a.py", name="alpha", content="alpha function")]
    delta1 = idx1.add_tiles(tiles1)

    # Merge delta from idx1 into idx2
    changes = idx2.merge(delta1)
    assert changes == 1
    assert idx2.count == 1

def test_merge_idempotent():
    idx1 = CRDTIndex(replica_id="r1", dim=64)
    idx2 = CRDTIndex(replica_id="r2", dim=64)

    tiles1 = [Tile(id="a", type="function", path="a.py", name="alpha", content="alpha function")]
    delta1 = idx1.add_tiles(tiles1)

    idx2.merge(delta1)
    changes = idx2.merge(delta1)  # again
    assert changes == 0  # idempotent
    assert idx2.count == 1

def test_merge_relevance():
    idx = CRDTIndex(dim=64)
    delta = Delta(relevance={"tile-x": 5})
    idx.merge(delta)
    assert idx._relevance["tile-x"] == 5


# ─── CRDT Semantic Dedup ─────────────────────────────────────────

def test_semantic_dedup_exact_content():
    idx = CRDTIndex(dim=64)
    tiles = [
        Tile(id="orig", type="function", path="a.py", name="foo", content="identical content here"),
    ]
    idx.add_tiles(tiles)

    # Simulate finding a semantic match
    new_tile = Tile(id="dup", type="function", path="b.py", name="foo2", content="identical content here")
    vec = idx.embedder.embed(f"{new_tile.name}\n{new_tile.content}")
    match = idx._find_semantic_match(new_tile, vec)
    assert match == "orig"


# ─── CRDT Save/Load ─────────────────────────────────────────────

def test_crdt_save_load_roundtrip(tmp_path):
    idx = CRDTIndex(replica_id="test-save", dim=64)
    tiles = [
        Tile(id="s1", type="function", path="x.py", name="save_func", content="save test content"),
        Tile(id="s2", type="class", path="y.py", name="SaveClass", content="save test class"),
    ]
    idx.add_tiles(tiles)
    idx.increment_relevance("s1", 7)

    fvt = str(tmp_path / "test.fvt")
    idx.save(fvt)

    # Load
    idx2 = CRDTIndex(dim=64)
    idx2.load(fvt)
    assert idx2.count == 2
    assert idx2._relevance.get("s1") == 7
    assert "s1" in idx2._tiles
    assert "s2" in idx2._tiles

def test_crdt_save_load_preserves_search(tmp_path):
    idx = CRDTIndex(replica_id="search-test", dim=64)
    tiles = [
        Tile(id="q1", type="function", path="a.py", name="query_func",
             content="query function for searching data"),
    ]
    idx.add_tiles(tiles)

    fvt = str(tmp_path / "test.fvt")
    idx.save(fvt)

    idx2 = CRDTIndex(dim=64)
    idx2.load(fvt)
    results = idx2.search("query search", top_k=1)
    assert len(results) > 0
    assert results[0].tile.name == "query_func"
