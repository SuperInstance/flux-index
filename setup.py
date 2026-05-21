"""Install script for flux-index."""
from setuptools import setup, find_packages

setup(
    name="flux-index",
    version="0.1.0",
    description="Semantic code search, zero dependencies",
    packages=find_packages(include=["flux_index", "flux_index.*"]),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "flux-index = flux_index.cli:main",
        ],
    },
)
