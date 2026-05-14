#!/usr/bin/env python3
"""
flux-index — Semantic code search, zero dependencies.

Usage:
    flux-index .                          # index current directory
    flux-index /path/to/repo              # index a repo
    flux-index search "auth flow"         # search indexed repo
    flux-index search --all "auth flow"   # search all known repos
    flux-index map                        # show codebase map
"""

import sys, os, argparse
from pathlib import Path

from flux_index.core import Index, Tile, SearchResult, extract_repo, index_repo


def _find_index(path="."):
    """Find .flux.fvt in or above current directory."""
    p = Path(path).resolve()
    while p != p.parent:
        fvt = p / ".flux.fvt"
        if fvt.exists():
            return str(fvt)
        p = p.parent
    return None


def _find_all_indexes():
    """Find all indexed repos."""
    indexes = []
    # Home index directory
    home = Path.home() / ".flux-index"
    if home.exists():
        indexes.extend(str(f) for f in home.glob("*.fvt"))
    # Walk up from cwd
    idx = _find_index()
    if idx and idx not in indexes:
        indexes.append(idx)
    # Scan /tmp for other indexed repos
    for fvt in Path("/tmp").rglob(".flux.fvt"):
        if str(fvt) not in indexes:
            indexes.append(str(fvt))
    return indexes


def _print_results(results: list, repo_name: str = "", max_width: int = 90):
    """Pretty-print search results."""
    if not results:
        print("  No results.")
        return
    
    prefix = f"  {repo_name}/" if repo_name else "  /"
    for r in results:
        t = r.tile
        # Format: [score] type: name (path:line)
        loc = f"{t.path}"
        if t.line:
            loc += f":{t.line}"
        header = f"[{r.score:.3f}] {t.type}: {t.name} ({loc})"
        
        # Show a one-line snippet from content (first non-empty line after declaration)
        snippet = ""
        for line in t.content.split("\n")[1:4]:
            stripped = line.strip()
            if stripped and not stripped.startswith(('"""', "'''")):
                snippet = stripped[:max_width]
                break
        
        print(f"{prefix}{header}")
        if snippet:
            print(f"{' ' * len(prefix)}  {snippet}")


def cmd_index(args):
    path = os.path.abspath(args.path)
    if not os.path.exists(path):
        print(f"Error: {path} does not exist")
        return 1
    
    output = args.output or os.path.join(path, ".flux.fvt")
    
    print(f"📂 Indexing {path}...")
    stats = index_repo(path, output, dim=args.dim, max_commits=args.commits)
    
    print(f"\n  ✅ {stats['tiles']} tiles indexed")
    for t, c in sorted(stats["types"].items(), key=lambda x: -x[1]):
        print(f"     {t}: {c}")
    if stats["languages"]:
        langs = ", ".join(f"{k} ({v})" for k, v in sorted(stats["languages"].items(), key=lambda x: -x[1]))
        print(f"     Languages: {langs}")
    print(f"     Size: {stats['size_kb']:.0f} KB → {stats['output']}")
    print(f"\n  🔍 Search: flux-index search \"your query here\"")
    return 0


def cmd_search(args):
    query = " ".join(args.query)
    
    if args.all:
        indexes = _find_all_indexes()
        if not indexes:
            print("No indexed repos found. Run 'flux-index .' first.")
            return 1
    else:
        idx = _find_index()
        if not idx:
            print("No index found. Run 'flux-index .' first.")
            return 1
        indexes = [idx]
    
    total_results = 0
    for fvt_path in indexes:
        repo_name = Path(fvt_path).stem.replace(".flux", "")
        idx = Index()
        try:
            idx.load(fvt_path)
        except Exception as e:
            print(f"  ⚠ {repo_name}: failed to load ({e})")
            continue
        
        results = idx.search(query, top_k=args.top, min_score=0.3)
        if results:
            print(f"\n  📦 {repo_name}/")
            _print_results(results, max_width=80)
            total_results += len(results)
    
    if total_results == 0:
        print(f"\n  No results for \"{query}\"")
        print("  Try broader terms, or index more repos with 'flux-index <path>'")
    
    return 0


def cmd_map(args):
    idx = _find_index()
    if not idx:
        print("No index found. Run 'flux-index .' first.")
        return 1
    
    index = Index()
    index.load(idx)
    
    repo_name = Path(idx).parent.name
    print(f"📦 {repo_name}")
    print(f"   Tiles: {index.count}")
    
    types = Counter(t.type for t in index.tiles)
    for t, c in types.most_common():
        bar = "█" * min(c, 40)
        print(f"   {t:10s} {c:4d} {bar}")
    
    # Show top functions/classes by name
    functions = [t for t in index.tiles if t.type in ("function", "class", "struct")]
    if functions:
        print(f"\n   Key symbols:")
        for t in functions[:20]:
            print(f"     {t.type}: {t.name} ({t.path}:{t.line})")
        if len(functions) > 20:
            print(f"     ... and {len(functions) - 20} more")
    
    return 0


def cmd_similar(args):
    """Find code similar to a given tile."""
    idx_path = _find_index()
    if not idx_path:
        print("No index found.")
        return 1
    
    index = Index()
    index.load(idx_path)
    query = " ".join(args.query)
    
    # First search for the target
    results = index.search(query, top_k=1)
    if not results:
        print(f"Nothing found for \"{query}\"")
        return 1
    
    target = results[0].tile
    print(f"Finding code similar to: {target.type}: {target.name} ({target.path}:{target.line})")
    
    # Search using the target's content as query
    similar = index.search(target.content, top_k=args.top + 1)
    for r in similar:
        if r.tile.id != target.id:
            print(f"  [{r.score:.3f}] {r.tile.type}: {r.tile.name} ({r.tile.path}:{r.tile.line})")
    
    return 0


from collections import Counter


def main():
    parser = argparse.ArgumentParser(
        prog="flux-index",
        description="flux-index — Semantic code search, zero dependencies",
    )
    sub = parser.add_subparsers(dest="command")
    
    # index (default if just a path)
    p_idx = sub.add_parser("index", help="Index a repository")
    p_idx.add_argument("path", help="Path to repository")
    p_idx.add_argument("-o", "--output", help="Output .fvt path")
    p_idx.add_argument("--dim", type=int, default=128, help="Embedding dimensions")
    p_idx.add_argument("--commits", type=int, default=200, help="Max commits to index")
    p_idx.set_defaults(func=cmd_index)
    
    # search
    p_search = sub.add_parser("search", help="Search indexed code")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("--all", action="store_true", help="Search all indexed repos")
    p_search.add_argument("--top", type=int, default=5, help="Results per repo")
    p_search.set_defaults(func=cmd_search)
    
    # map
    p_map = sub.add_parser("map", help="Show codebase map")
    p_map.set_defaults(func=cmd_map)
    
    # similar
    p_sim = sub.add_parser("similar", help="Find similar code")
    p_sim.add_argument("query", nargs="+", help="Reference query")
    p_sim.add_argument("--top", type=int, default=5, help="Number of results")
    p_sim.set_defaults(func=cmd_similar)
    
    # Default: if first arg is a directory, index it
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        args = argparse.Namespace(
            command="index", path=sys.argv[1],
            output=None, commits=200, dim=128, func=cmd_index,
        )
    else:
        args = parser.parse_args()
    
    if hasattr(args, "func"):
        return args.func(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
