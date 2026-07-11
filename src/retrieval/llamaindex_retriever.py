"""
src/retrieval/llamaindex_retriever.py — LlamaIndex retriever wrapper.

Builds a VectorStoreIndex from the configured vector store and
retrieves top-k nodes above the similarity cutoff.
"""

from __future__ import annotations

import structlog
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from src.config import cfg

logger = structlog.get_logger(__name__)


class LlamaIndexRetriever:
    """Retrieves relevant document chunks from a vector store collection."""

    def __init__(self, collection: str | None = None) -> None:
        from src.config import settings

        self.collection = collection or settings.chroma_collection
        self._index: VectorStoreIndex | None = None

    def _get_index(self) -> VectorStoreIndex:
        if self._index is None:
            from src.vectordb.factory import get_vector_store

            vector_store = get_vector_store(self.collection)
            self._index = VectorStoreIndex.from_vector_store(vector_store)
            logger.debug("retriever_index_built", collection=self.collection)
        return self._index

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieve top-k nodes above similarity cutoff."""
        retrieval_cfg = cfg._data.get("retrieval", {}) if hasattr(cfg, "_data") else {}
        top_k: int = retrieval_cfg.get("top_k", 5)
        cutoff: float = retrieval_cfg.get("similarity_cutoff", 0.7)

        index = self._get_index()
        retriever = index.as_retriever(
            similarity_top_k=top_k,
        )

        nodes = retriever.retrieve(query)

        # Filter by similarity cutoff
        filtered = [n for n in nodes if (n.score or 0.0) >= cutoff]

        logger.info(
            "retrieval_complete",
            collection=self.collection,
            query=query[:80],
            raw_count=len(nodes),
            filtered_count=len(filtered),
            cutoff=cutoff,
        )

        return filtered
