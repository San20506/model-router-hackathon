"""Shared pytest fixtures and hooks.

The test suite runs against the zero-dependency Dice embedder by default so it
doesn't download sentence-transformers models or wait on GPU/CPU encoding.
Individual tests can still opt into MiniLM explicitly if they need semantic
similarity.
"""

import pytest


def pytest_configure(config):
    """Force Dice embedder for all tests by default."""
    import os
    # Set before any SourceOfTruth is imported/instantiated
    os.environ.setdefault("MODEL_ROUTER_EMBEDDER", "dice")


@pytest.fixture
def use_minilm():
    """Temporarily allow a test to use the MiniLM embedder."""
    import os
    old = os.environ.get("MODEL_ROUTER_EMBEDDER")
    os.environ["MODEL_ROUTER_EMBEDDER"] = "minilm"
    yield
    if old is None:
        os.environ.pop("MODEL_ROUTER_EMBEDDER", None)
    else:
        os.environ["MODEL_ROUTER_EMBEDDER"] = old
