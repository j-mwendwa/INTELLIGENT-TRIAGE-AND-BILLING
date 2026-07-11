"""
src/vectordb/qdrant_store.py
============================
Qdrant vector store factory for the Intelligent Triage & Billing RAG system.

Responsibilities
----------------
* Build a :class:`qdrant_client.QdrantClient` from application settings.
* Ensure the target collection exists (creating it with sensible defaults if
  it does not).
* Return a LlamaIndex-compatible :class:`QdrantVectorStore` with hybrid search
  enabled when the Qdrant server supports it.
"""

from __future__ import annotations

import logging

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

_DISTANCE = rest.Distance.COSINE


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int = 384,
) -> None:
    """
    Ensure *collection* exists in Qdrant.

    If the collection is already present its configuration is left untouched.
    Otherwise a new collection is created with:

    * **distance** : cosine similarity
    * **vector size** : *vector_size* (default 384, matching
      ``sentence-transformers/all-MiniLM-L6-v2``)
    * **on_disk_payload** : ``True`` – keeps large metadata on disk to reduce
      RAM pressure in production.

    Parameters
    ----------
    client:
        An active :class:`~qdrant_client.QdrantClient` instance.
    collection:
        The collection name to verify or create.
    vector_size:
        Dimensionality of the dense embedding vectors stored in the collection.
    """
    try:
        info = client.get_collection(collection)
        logger.info(
            "Qdrant collection already exists: name=%r, vectors=%d",
            collection,
            info.vectors_count or 0,
        )
    except (UnexpectedResponse, Exception) as exc:  # noqa: BLE001
        # Qdrant raises UnexpectedResponse (404) when the collection is missing.
        logger.info(
            "Collection %r not found (%s). Creating it now …",
            collection,
            exc,
        )
        client.create_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=_DISTANCE,
            ),
            on_disk_payload=True,
        )
        logger.info(
            "Qdrant collection created: name=%r, size=%d, distance=%s",
            collection,
            vector_size,
            _DISTANCE,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_qdrant_store(collection: str) -> QdrantVectorStore:
    """
    Return a :class:`QdrantVectorStore` for the given *collection* name.

    Steps
    -----
    1. Build a :class:`~qdrant_client.QdrantClient` pointed at
       ``settings.qdrant_url``, authenticated with ``settings.qdrant_api_key``
       (``None`` for unauthenticated / local deployments).
    2. Call :func:`ensure_collection` so the collection always exists before
       the store is handed back to the caller.
    3. Construct the LlamaIndex store with ``enable_hybrid=True`` to activate
       sparse + dense retrieval; LlamaIndex silently ignores the flag when the
       Qdrant version does not support hybrid search.

    Parameters
    ----------
    collection:
        The Qdrant collection name to use (or create) for this store.

    Returns
    -------
    QdrantVectorStore
        A LlamaIndex-compatible vector store wrapping the resolved Qdrant
        collection.
    """
    api_key: str | None = settings.qdrant_api_key or None

    logger.info(
        "Connecting to Qdrant: url=%s, authenticated=%s",
        settings.qdrant_url,
        api_key is not None,
    )

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key,
    )

    ensure_collection(client, collection, vector_size=384)

    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        enable_hybrid=True,
    )

    logger.info(
        "QdrantVectorStore ready: collection=%r, hybrid=True",
        collection,
    )
    return store
