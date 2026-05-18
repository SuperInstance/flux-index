"""Tests for flux-index core functionality."""

import os, tempfile, pathlib, subprocess, sys

def test_cli_install():
    """Verify flux-index CLI is importable."""
    from flux_index.cli import main
    assert callable(main)

def test_core_import():
    """Verify core modules import."""
    from flux_index import core, extractor
    assert hasattr(core, 'Index')
    assert hasattr(extractor, 'repo_to_vectors')

def test_extractor_no_broken_deps():
    """
    Verify extractor has zero external dependencies.
    This is the critical test — extractor.py should NOT import
    flux_vector_twin or local_plato.
    """
    import ast
    import importlib.util
    
    spec = importlib.util.find_spec("flux_index.extractor")
    assert spec is not None, "extractor module must be importable"
    
    with open(spec.origin) as f:
        tree = ast.parse(f.read())
    
    forbidden = {"flux_vector_twin", "local_plato"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                assert name not in forbidden, \
                    f"extractor.py imports forbidden: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split(".")[0]
                assert name not in forbidden, \
                    f"extractor.py imports forbidden: {node.module}"

def test_index_small_repo(tmp_path):
    """Index a tiny repo via core.index_repo()."""
    from flux_index.core import index_repo
    
    (tmp_path / "hello.py").write_text("def hello(): return 'world'\n")
    (tmp_path / "README.md").write_text("Hello world project\n")
    
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(tmp_path), capture_output=True)
    
    output = str(tmp_path / ".flux.fvt")
    stats = index_repo(str(tmp_path), output=output, dim=64)
    assert stats["tiles"] > 0, "No tiles indexed"
    assert os.path.exists(output), ".flux.fvt not created"

def test_search_after_index(tmp_path):
    """Index a repo, then search it via API."""
    from flux_index.core import index_repo, Index
    
    src = tmp_path / "auth.py"
    src.write_text("""
def authenticate_user(username: str, password: str) -> bool:
    # Authenticate a user against the database.
    return username == "admin" and password == "secret"

def logout_user(user_id: int) -> None:
    # Log out a user by clearing their session.
    pass

class SessionManager:
    # Manages user sessions.
    def create_session(self, user_id: int) -> str:
        return "token_" + str(user_id)
""")
    
    (tmp_path / "README.md").write_text("Authentication service\n")
    
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add auth module"], cwd=str(tmp_path), capture_output=True)
    
    output = str(tmp_path / ".flux.fvt")
    stats = index_repo(str(tmp_path), output=output, dim=64)
    assert stats["tiles"] > 0
    
    idx = Index()
    idx.load(output)
    results = idx.search("user login authentication", top_k=3)
    assert len(results) > 0, "Expected results from auth code"
    names = [r.tile.name for r in results]
    found_auth = any(
        "authenticate" in n.lower() or "session" in n.lower() or "login" in n.lower()
        for n in names
    )
    assert found_auth, f"Expected auth-related results, got: {names}"

def test_core_extract_py():
    """Test that Python extraction in core works."""
    from flux_index.core import extract_py
    
    src = "def greet(name): return f'Hello {name}'\nclass Calc:\n    def add(self, a, b): return a + b\n"
    tiles = extract_py("test.py", src)
    
    names = [t.name for t in tiles]
    assert "greet" in names, f"Expected 'greet' in {names}"
    
    calc = [t for t in tiles if t.name == "Calc"]
    assert len(calc) == 1
    assert calc[0].type == "class"

def test_embedder():
    """Test embedder determinism."""
    from flux_index.core import Embedder
    
    e = Embedder(64)
    e.train(["hello world", "goodbye world"])
    
    v1 = e.embed("hello")
    v2 = e.embed("hello")
    assert v1 == v2, "Embedding not deterministic"
    
    v3 = e.embed("goodbye")
    assert v1 != v3, "Different inputs should differ"

def test_index_search():
    """Test core Index add and search."""
    from flux_index.core import Index, Tile
    
    idx = Index(64)
    tiles = [
        Tile(id="1", type="function", path="auth.py", name="login",
             content="handles user login and password verification"),
        Tile(id="2", type="function", path="db.py", name="connect",
             content="database connection pool management"),
        Tile(id="3", type="function", path="pay.py", name="checkout",
             content="payment processing and refund handling"),
    ]
    idx.add(tiles)
    
    results = idx.search("user authentication", top_k=3)
    assert len(results) > 0
    assert results[0].tile.type == "function"
    for r in results:
        assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"

def test_extract_file_tiles():
    """Test that extractor's file tile extraction works."""
    from flux_index.extractor import extract_file_tiles
    
    src = """
def greet(name: str) -> str:
    # Return greeting.
    return f"Hello {name}"

class Calculator:
    # Basic calculator.
    def add(self, a, b): return a + b
"""
    
    tiles = extract_file_tiles("test.py", src)
    assert len(tiles) > 0
    names = [t.name for t in tiles]
    assert "greet" in names, f"Expected greet in {names}"
    assert "Calculator" in names, f"Expected Calculator in {names}"

def test_search_repo_from_extractor(tmp_path):
    """Test that extractor's search_repo works."""
    from flux_index.extractor import repo_to_vectors, search_repo
    
    (tmp_path / "hello.py").write_text("def hello(): return 'world'\n")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(tmp_path), capture_output=True)
    
    output = str(tmp_path / "test.fvt")
    stats = repo_to_vectors(str(tmp_path), output_path=output, dim=64)
    assert stats["tiles_extracted"] > 0
    assert os.path.exists(output)
    
    results = search_repo(output, "hello", top_k=5)
    assert len(results) > 0, "No search results"

def test_load_save_roundtrip(tmp_path):
    """Test index save/load roundtrip."""
    from flux_index.core import Index, Tile
    
    orig = Index(64)
    tiles = [
        Tile(id="a1", type="function", path="test.py", name="alpha", content="first function"),
        Tile(id="b2", type="class", path="test.py", name="Beta", content="some class"),
    ]
    orig.add(tiles)
    
    fvt = str(tmp_path / "test.fvt")
    orig.save(fvt)
    
    loaded = Index()
    count = loaded.load(fvt)
    assert count == 2
    assert {t.name for t in loaded.tiles} == {"alpha", "Beta"}
