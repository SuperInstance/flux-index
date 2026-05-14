"""
flux-index — Semantic code search, zero dependencies.
"""

__version__ = "0.1.0"

from flux_index.core import Index, Tile, SearchResult, extract_repo, index_repo

__all__ = ["Index", "Tile", "SearchResult", "extract_repo", "index_repo"]
