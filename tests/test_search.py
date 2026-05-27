"""Tests for flux-index search module (FluxVectorTwin)."""

import os, tempfile, json
from collections import Counter
from flux_index.search import (
    ngrams, word_features, char_features, hash_feature,
    text_to_embedding, VectorEntry, FluxVectorTwin,
)


# ─── ngrams ───────────────────────────────────────────────────────

def test_ngrams_basic():
    result = ngrams("abc", 3)
    assert "^abc" in result or "abc" in "".join(result)

def test_ngrams_short():
    result = ngrams("ab", 2)
    assert len(result) > 0

def test_ngrams_empty():
    result = ngrams("", 3)
    assert result == []


# ─── word_features ────────────────────────────────────────────────

def test_word_features_basic():
    wf = word_features("hello world hello")
    assert wf["hello"] == 2
    assert wf["world"] == 1

def test_word_features_empty():
    assert word_features("") == Counter()


# ─── char_features ────────────────────────────────────────────────

def test_char_features_basic():
    cf = char_features("abc", ns=(2,))
    assert len(cf) > 0

def test_char_features_multiple_ns():
    cf = char_features("hello", ns=(2, 3))
    assert len(cf) > 5  # multiple n-gram sizes

def test_char_features_empty():
    result = char_features("")
    # Empty string still produces boundary n-grams (^$), just verify it doesn't crash
    assert isinstance(result, Counter)


# ─── hash_feature ─────────────────────────────────────────────────

def test_hash_feature_in_range():
    for feat in ["abc", "xyz", "w:hello", "b:ab"]:
        idx = hash_feature(feat, dim=128)
        assert 0 <= idx < 128

def test_hash_feature_deterministic():
    assert hash_feature("test", dim=64) == hash_feature("test", dim=64)


# ─── text_to_embedding ───────────────────────────────────────────

def test_embedding_dimension():
    vec = text_to_embedding("hello world", dim=64)
    assert len(vec) == 64

def test_embedding_normalized():
    vec = text_to_embedding("some text here", dim=128)
    mag = sum(x * x for x in vec) ** 0.5
    assert abs(mag - 1.0) < 1e-6 or mag == 0.0

def test_embedding_deterministic():
    v1 = text_to_embedding("test input", dim=64)
    v2 = text_to_embedding("test input", dim=64)
    assert v1 == v2

def test_embedding_with_idf():
    idf = {"hel": 2.0, "w:hello": 3.0}
    vec = text_to_embedding("hello", idf_weights=idf, dim=64)
    assert len(vec) == 64

def test_embedding_different_inputs():
    v1 = text_to_embedding("authentication", dim=64)
    v2 = text_to_embedding("database connection", dim=64)
    assert v1 != v2


# ─── FluxVectorTwin basics ───────────────────────────────────────

class FakeTile:
    """Minimal tile object for search.py index_tiles compatibility."""
    def __init__(self, tile_id, room, question, answer, timestamp=0.0):
        self.tile_id = tile_id
        self.room = room
        self.question = question
        self.answer = answer
        self.timestamp = timestamp


def test_fvt_empty():
    fvt = FluxVectorTwin(dim=64)
    assert fvt.count == 0
    assert "untrained" in fvt.stats()

def test_fvt_train():
    fvt = FluxVectorTwin(dim=64)
    fvt.train(["hello world", "foo bar baz"])
    assert fvt._trained
    assert len(fvt.idf_weights) > 0

def test_fvt_index_tiles():
    fvt = FluxVectorTwin(dim=64)
    tiles = [
        FakeTile("t1", "python", "How to sort a list?", "Use sorted()"),
        FakeTile("t2", "python", "How to read a file?", "Use open()"),
    ]
    count = fvt.index_tiles(tiles)
    assert count == 2
    assert fvt.count == 2

def test_fvt_search():
    fvt = FluxVectorTwin(dim=64)
    tiles = [
        FakeTile("t1", "python", "How to sort a list?", "Use sorted() function"),
        FakeTile("t2", "rust", "How to sort a vector?", "Use .sort() method"),
        FakeTile("t3", "python", "How to read a file?", "Use open() built-in"),
    ]
    fvt.index_tiles(tiles)
    results = fvt.search("sort list", top_k=2)
    assert len(results) > 0
    assert len(results) <= 2
    for entry, score in results:
        assert isinstance(entry, VectorEntry)
        assert 0.0 <= score

def test_fvt_search_room():
    fvt = FluxVectorTwin(dim=64)
    tiles = [
        FakeTile("t1", "python", "How to sort?", "sorted()"),
        FakeTile("t2", "rust", "How to sort?", "vec.sort()"),
        FakeTile("t3", "python", "How to read?", "open()"),
    ]
    fvt.index_tiles(tiles)
    results = fvt.search_room("sort", "python", top_k=2)
    for entry, score in results:
        assert entry.room == "python"

def test_fvt_similar_to():
    fvt = FluxVectorTwin(dim=64)
    tiles = [
        FakeTile("t1", "python", "How to sort?", "sorted()"),
        FakeTile("t2", "python", "How to sort a list?", "sorted(lst)"),
        FakeTile("t3", "python", "How to read a file?", "open()"),
    ]
    fvt.index_tiles(tiles)
    results = fvt.similar_to("t1", top_k=2)
    assert len(results) > 0
    # t1 should not appear in its own similar results
    for entry, score in results:
        assert entry.tile_id != "t1"

def test_fvt_similar_to_missing():
    fvt = FluxVectorTwin(dim=64)
    tiles = [FakeTile("t1", "r", "q", "a")]
    fvt.index_tiles(tiles)
    assert fvt.similar_to("nonexistent", top_k=5) == []

def test_fvt_search_empty():
    fvt = FluxVectorTwin(dim=64)
    assert fvt.search("anything", top_k=5) == []


# ─── FluxVectorTwin Save/Load ────────────────────────────────────

def test_fvt_save_load_roundtrip(tmp_path):
    fvt = FluxVectorTwin(dim=64)
    tiles = [
        FakeTile("t1", "room1", "What is Python?", "A programming language", timestamp=1000.0),
        FakeTile("t2", "room2", "What is Rust?", "A systems language", timestamp=2000.0),
    ]
    fvt.index_tiles(tiles)

    path = str(tmp_path / "test_store.json")
    fvt.save(path)
    assert os.path.exists(path)

    fvt2 = FluxVectorTwin(dim=64)
    count = fvt2.load(path)
    assert count == 2
    assert fvt2.count == 2
    assert fvt2._trained
    assert fvt2.entries[0].tile_id == "t1"

def test_fvt_stats():
    fvt = FluxVectorTwin(dim=64)
    tiles = [FakeTile("t1", "r", "q", "a")]
    fvt.index_tiles(tiles)
    stats = fvt.stats()
    assert "1 tiles" in stats
    assert "trained" in stats


# ─── Cosine utility ──────────────────────────────────────────────

def test_cosine_similarity():
    a = [1.0, 0.0]
    b = [1.0, 0.0]
    assert abs(FluxVectorTwin._cosine(a, b) - 1.0) < 1e-6

def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(FluxVectorTwin._cosine(a, b)) < 1e-6

def test_cosine_zero_vector():
    assert FluxVectorTwin._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
