"""
flux-index — Semantic code search, zero dependencies.

Spring-load any repo into a searchable vector space.
"""

__version__ = "0.1.0"

from flux_index.extractor import repo_to_vectors, search_repo
from flux_index.search import FluxVectorTwin

__all__ = ["repo_to_vectors", "search_repo", "FluxVectorTwin"]
