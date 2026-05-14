#!/usr/bin/env python3
"""
flux-index CLI — semantic code search from the command line.

Usage:
    flux-index .                          # index current directory
    flux-index /path/to/repo              # index a repo
    flux-index search "auth flow"         # search indexed repo
    flux-index search --all "auth flow"   # search all indexed repos
    flux-index map                        # show codebase map
    flux-index status                     # show index status
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flux_index.extractor import repo_to_vectors, search_repo


def find_index(path="."):
    """Find .flux.fvt file in or above current directory."""
    p = Path(path).resolve()
    while p != p.parent:
        fvt = p / ".flux.fvt"
        if fvt.exists():
            return str(fvt)
        p = p.parent
    return None


def find_all_indexes():
    """Find all .flux.fvt files in common locations."""
    indexes = []
    # Check ~/.flux-index/
    home_idx = Path.home() / ".flux-index"
    if home_idx.exists():
        for fvt in home_idx.glob("*.fvt"):
            indexes.append(str(fvt))
    # Check current directory tree
    for fvt in Path(".").rglob(".flux.fvt"):
        indexes.append(str(fvt))
    return indexes


def cmd_index(args):
    """Index a repository."""
    path = args.path
    if not os.path.exists(path):
        print(f"Error: {path} does not exist")
        return 1
    
    output = args.output or os.path.join(path, ".flux.fvt")
    
    print(f"Indexing {path}...")
    stats = repo_to_vectors(
        path,
        output_path=output,
        dim=args.dim,
        max_commits=args.commits,
    )
    
    print(f"\n  Tiles extracted: {stats['tiles_extracted']}")
    print(f"  Types: {stats['type_breakdown']}")
    if stats['language_breakdown']:
        print(f"  Languages: {stats['language_breakdown']}")
    print(f"  Index size: {stats['output_size_kb']:.0f} KB")
    print(f"  Saved to: {stats['output_path']}")
    print(f"\n  Search it: flux-index search \"your query here\"")
    return 0


def cmd_search(args):
    """Search an indexed repository."""
    if args.all:
        indexes = find_all_indexes()
        if not indexes:
            print("No indexed repos found. Run 'flux-index .' first.")
            return 1
    else:
        idx = find_index()
        if not idx:
            # Try current directory name in ~/.flux-index/
            name = Path(".").resolve().name
            fallback = str(Path.home() / ".flux-index" / f"{name}.fvt")
            if os.path.exists(fallback):
                idx = fallback
            else:
                print("No index found. Run 'flux-index .' first.")
                return 1
        indexes = [idx]
    
    query = " ".join(args.query)
    
    for fvt_path in indexes:
        repo_name = Path(fvt_path).stem.replace(".flux", "")
        try:
            results = search_repo(fvt_path, query, top_k=args.top)
        except Exception as e:
            print(f"  {repo_name}: error loading index ({e})")
            continue
        
        if results:
            print(f"\n  {repo_name}/")
            for entry, score in results:
                # Truncate snippet for display
                snippet = entry.snippet.replace("\n", " ")[:100]
                print(f"    [{score:.3f}] {snippet}")
    
    return 0


def cmd_map(args):
    """Show a map of the indexed codebase."""
    idx = find_index()
    if not idx:
        print("No index found. Run 'flux-index .' first.")
        return 1
    
    from flux_index.search import FluxVectorTwin
    twin = FluxVectorTwin()
    twin.load(idx)
    
    print(f"Codebase: {Path(idx).parent.name}")
    print(f"Tiles: {twin.count}")
    print(f"Dimensions: {twin.dim}")
    print()
    
    # Group by type
    from collections import Counter
    types = Counter()
    for entry in twin.entries:
        # Extract type from snippet
        if entry.snippet.startswith("function:"):
            types["function"] += 1
        elif entry.snippet.startswith("class:"):
            types["class"] += 1
        elif entry.snippet.startswith("struct:"):
            types["struct"] += 1
        elif entry.snippet.startswith("commit:"):
            types["commit"] += 1
        elif entry.snippet.startswith("readme:"):
            types["readme"] += 1
        else:
            types["file"] += 1
    
    print("Breakdown:")
    for t, c in types.most_common():
        print(f"  {t}: {c}")
    
    return 0


def cmd_status(args):
    """Show index status."""
    idx = find_index()
    if idx:
        size = os.path.getsize(idx) / 1024
        print(f"Index: {idx} ({size:.0f} KB)")
    else:
        print("No index in current directory tree.")
    
    # Check home indexes
    home_idx = Path.home() / ".flux-index"
    if home_idx.exists():
        fvts = list(home_idx.glob("*.fvt"))
        if fvts:
            print(f"\nHome indexes ({len(fvts)}):")
            for fvt in fvts:
                size = os.path.getsize(fvt) / 1024
                print(f"  {fvt.stem}: {size:.0f} KB")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="flux-index — Semantic code search, zero dependencies",
        prog="flux-index",
    )
    subparsers = parser.add_subparsers(dest="command")
    
    # index
    p_index = subparsers.add_parser("index", help="Index a repository")
    p_index.add_argument("path", help="Path to repository")
    p_index.add_argument("-o", "--output", help="Output .fvt file path")
    p_index.add_argument("--dim", type=int, default=64, help="Embedding dimensions")
    p_index.add_argument("--commits", type=int, default=200, help="Max commits to index")
    p_index.set_defaults(func=cmd_index)
    
    # search
    p_search = subparsers.add_parser("search", help="Search indexed repository")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("--all", action="store_true", help="Search all indexed repos")
    p_search.add_argument("--top", type=int, default=10, help="Number of results")
    p_search.set_defaults(func=cmd_search)
    
    # map
    p_map = subparsers.add_parser("map", help="Show codebase map")
    p_map.set_defaults(func=cmd_map)
    
    # status
    p_status = subparsers.add_parser("status", help="Show index status")
    p_status.set_defaults(func=cmd_status)
    
    # Default: if just a path, index it
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in ("search", "map", "status", "index"):
        # Treat as: flux-index /path/to/repo
        args = argparse.Namespace(
            command="index", path=sys.argv[1],
            output=None, dim=64, commits=200, func=cmd_index,
        )
    else:
        args = parser.parse_args()
    
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
