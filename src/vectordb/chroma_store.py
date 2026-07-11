"""
src/vectordb/chroma_store.py
============================
Chroma vector store factory for the Intelligent Triage & Billing RAG system.

Attempts to connect to a remote Chroma HTTP server first; falls back to a local
PersistentClient when the HTTP server is unreachable.
"""

from __future__ import annotations

import logging

import chromadb
from chromadb import HttpClient, PersistentClient
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import CHROMA_DIR, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_http_client(host: str, port: int) -> chromadb.ClientAPI | None:
    """
    Attempt to connect to a Chroma HTTP server.

    Returns the :class:`chromadb.ClientAPI` instance on success, or ``None``
    if the server is unreachable (any :class:`Exception` is treated as a
    connection failure so that the caller can fall back gracefully).
    """
    try:
        client: chromadb.ClientAPI = HttpClient(host=host, port=port)
        # Heartbeat verifies that the server is actually responding.
        client.heartbeat()
        logger.info(
            "Chroma HTTP client connected → %s:%s",
            host,
            port,
        )
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Chroma HTTP server at %s:%s is unreachable (%s). Falling back to PersistentClient.",
            host,
            port,
            exc,
        )
        return None


def _get_persistent_client() -> chromadb.ClientAPI:
    """
    Create a local Chroma :class:`PersistentClient` backed by *CHROMA_DIR*.
    """
    path = str(CHROMA_DIR)
    logger.info("Chroma PersistentClient → path=%s", path)
    return PersistentClient(path=path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_chroma_store(collection: str) -> ChromaVectorStore:
    """
    Return a :class:`ChromaVectorStore` for the given *collection* name.

    Resolution order
    ----------------
    1. Try to reach a Chroma HTTP server at
       ``settings.chroma_host:settings.chroma_port``.
    2. If the HTTP server is unavailable, fall back to a
       :class:`PersistentClient` rooted at ``CHROMA_DIR``.

    Parameters
    ----------
    collection:
        The Chroma collection name to use (or create) for this store.

    Returns
    -------
    ChromaVectorStore
        A LlamaIndex-compatible vector store wrapping the resolved Chroma
        collection.
    """
    # --- resolve the chromadb client ---
    client: chromadb.ClientAPI = (
        _try_http_client(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        or _get_persistent_client()
    )

    # --- obtain (or create) the collection ---
    chroma_collection = client.get_or_create_collection(collection)
    logger.info(
        "Chroma collection resolved: name=%r, count=%d",
        chroma_collection.name,
        chroma_collection.count(),
    )

    return ChromaVectorStore(chroma_collection=chroma_collection)
