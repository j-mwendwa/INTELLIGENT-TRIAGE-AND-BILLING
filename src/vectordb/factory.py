"""
src/vectordb/factory.py
=======================
Central vector store factory for the Intelligent Triage & Billing RAG system.

``get_vector_store`` is the single entry-point used by the rest of the
application to obtain a LlamaIndex-compatible vector store.  The concrete
backend is selected via ``settings.vector_backend``; the factory is designed
so that adding a new backend requires only a new branch in the dispatch dict.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from src.config import settings
from src.core.exceptions import VectorStoreError

if TYPE_CHECKING:
    pass  # kept for future type-only imports

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

# Lazily import store constructors inside the factory so that only the
# dependencies for the *selected* backend need to be installed at runtime.
_BACKEND_LOADERS: dict[str, str] = {
    "chroma": "src.vectordb.chroma_store.get_chroma_store",
    "qdrant": "src.vectordb.qdrant_store.get_qdrant_store",
}


def _load_store_fn(dotted_path: str):
    """
    Import and return a callable identified by its dotted module path.

    Example
    -------
    ``_load_store_fn("src.vectordb.chroma_store.get_chroma_store")``
    returns the :func:`get_chroma_store` function without importing the whole
    module at module-load time.
    """
    module_path, _, attr = dotted_path.rpartition(".")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_vector_store(
    collection: str | None = None,
) -> BasePydanticVectorStore:
    """
    Return a LlamaIndex vector store for the requested *collection*.

    Parameters
    ----------
    collection:
        The collection (or index) name to use.  When ``None``, the value of
        ``settings.chroma_collection`` is used as the default, regardless of
        which backend is active.

    Returns
    -------
    BasePydanticVectorStore
        A fully initialised, backend-specific vector store instance.

    Raises
    ------
    VectorStoreError
        When ``settings.vector_backend`` does not match any known backend.
    """
    resolved_collection: str = collection or settings.chroma_collection
    backend: str = settings.vector_backend.lower()

    logger.info(
        "Resolving vector store: backend=%r, collection=%r",
        backend,
        resolved_collection,
    )

    loader_path = _BACKEND_LOADERS.get(backend)
    if loader_path is None:
        raise VectorStoreError(
            f"Unknown backend: {settings.vector_backend!r}. "
            f"Supported backends: {sorted(_BACKEND_LOADERS.keys())}"
        )

    store_fn = _load_store_fn(loader_path)
    store: BasePydanticVectorStore = store_fn(resolved_collection)

    logger.info(
        "Vector store ready: backend=%r, collection=%r, type=%s",
        backend,
        resolved_collection,
        type(store).__name__,
    )
    return store
