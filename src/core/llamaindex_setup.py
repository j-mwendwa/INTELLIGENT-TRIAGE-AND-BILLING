"""
src/core/llamaindex_setup.py
────────────────────────────
LlamaIndex global ``Settings`` configuration for the Intelligent Triage &
Billing RAG system.

Architecture decision
---------------------
This system uses a **split-responsibility** design:

* **LlamaIndex** handles document ingestion, chunking, embedding, and
  vector-store retrieval.
* **LangGraph** (via LangChain) handles LLM inference, chain-of-thought
  orchestration, and multi-step reasoning.

As a result, ``Settings.llm`` is intentionally set to ``None`` here.  If
LlamaIndex ever tries to call an LLM directly (e.g. in a synthesiser or
re-ranker), an explicit ``ValueError`` will be raised rather than
inadvertently spinning up a default OpenAI client.

Embeddings
----------
``BAAI/bge-small-en-v1.5`` is a compact (384-dimensional) English-language
bi-encoder that achieves MTEB leaderboard scores close to much larger models.
Running it locally (CPU or GPU) keeps embedding costs at zero and avoids
network round-trips during ingestion.

Usage
-----
    from src.core.llamaindex_setup import setup_llamaindex
    setup_llamaindex()           # call once during application start-up
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# ── Model constants ────────────────────────────────────────────────────────────

#: HuggingFace model ID for the sentence-transformer used in all embeddings.
EMBED_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

#: Output dimensionality of ``bge-small-en-v1.5``.
EMBED_DIM: int = 384

#: Maximum sequence length supported by the embedding model (in tokens).
EMBED_MAX_SEQ_LEN: int = 512

#: Batch size used when embedding large document corpora.
EMBED_BATCH_SIZE: int = 32


def setup_llamaindex(
    embed_model_name: str = EMBED_MODEL_NAME,
    embed_batch_size: int = EMBED_BATCH_SIZE,
    *,
    device: str = "cpu",
) -> None:
    """
    Configure LlamaIndex global ``Settings`` for this application.

    Must be called **once** at application start-up, before any LlamaIndex
    index or query-engine objects are created.

    Parameters
    ----------
    embed_model_name:
        HuggingFace model identifier for the sentence-transformer.
        Defaults to ``"BAAI/bge-small-en-v1.5"`` (384 dim).
    embed_batch_size:
        Number of text chunks encoded in a single forward pass.  Reduce this
        if you hit CUDA OOM errors during bulk ingestion.
    device:
        Torch device string: ``"cpu"``, ``"cuda"``, or ``"mps"``.
        ``"cpu"`` is the safe default for cloud deployments without a GPU.

    Side Effects
    ------------
    * Sets ``llama_index.core.Settings.embed_model`` to a
      ``HuggingFaceEmbedding`` instance loaded from *embed_model_name*.
    * Sets ``llama_index.core.Settings.llm`` to ``None`` (LLM inference is
      handled by LangGraph, not LlamaIndex).

    Raises
    ------
    ImportError
        If ``llama-index-embeddings-huggingface`` is not installed.
    RuntimeError
        If the model cannot be loaded (e.g. no network access and the model
        is not cached locally).
    """
    # Lazy imports keep startup fast when LlamaIndex is not needed.
    try:
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex HuggingFace integration is not installed.\n"
            "Run: pip install llama-index-embeddings-huggingface"
        ) from exc

    logger.info(
        "llamaindex_setup_start",
        embed_model=embed_model_name,
        embed_dim=EMBED_DIM,
        embed_batch_size=embed_batch_size,
        device=device,
    )

    # ── Embedding model ────────────────────────────────────────────────────────
    embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        max_length=EMBED_MAX_SEQ_LEN,
        embed_batch_size=embed_batch_size,
        device=device,
    )
    Settings.embed_model = embed_model

    # ── Disable LlamaIndex's built-in LLM ─────────────────────────────────────
    # LLM orchestration is fully delegated to LangGraph.  Setting this to
    # ``None`` surfaces any accidental LlamaIndex-initiated LLM call as a
    # clear error rather than a silent (and potentially expensive) API call.
    Settings.llm = None

    logger.info(
        "llamaindex_setup_complete",
        embed_model=embed_model_name,
        embed_dim=EMBED_DIM,
        llm="disabled (LangGraph handles LLM)",
        device=device,
    )
