"""
src/ingestion/llamaindex_pipeline.py
────────────────────────────────────
Ingestion pipeline using LlamaIndex for the Intelligent Triage & Billing RAG
system.

Flow
----
    path
      → _safe_file_list()          (path traversal prevention + format filter)
      → SimpleDirectoryReader      (LlamaIndex document loader)
      → IngestionPipeline          (SentenceSplitter ± TitleExtractor)
      → HuggingFaceEmbedding       (configured via setup_llamaindex)
      → VectorStore                (Chroma or Qdrant via get_vector_store)

Security
--------
``_ALLOWED_INGEST_ROOTS`` limits ingestion to directories under ``data/``.
``_safe_file_list`` uses ``os.walk(followlinks=False)`` to prevent symlink
escapes, and validates each resolved path against the allowed roots before
returning it.

Usage
-----
    from src.core.llamaindex_setup import setup_llamaindex
    from src.ingestion.llamaindex_pipeline import ingest_directory, load_index

    setup_llamaindex()                           # call once at startup
    n = ingest_directory("data/raw")             # returns node count
    index = load_index()                         # load existing index
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.config import DATA_DIR, cfg
from src.core.exceptions import IngestionError
from src.vectordb.factory import get_vector_store

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex

logger = structlog.get_logger(__name__)

# ── Security: allowed ingestion roots ────────────────────────────────────────
# All ingested files must reside under one of these resolved absolute paths.
# This prevents path-traversal attacks that could expose system files.
_ALLOWED_INGEST_ROOTS: tuple[Path, ...] = (DATA_DIR.resolve(),)

# ── Binary / non-indexable extension blocklist ────────────────────────────────
# These formats cannot be meaningfully chunked into text for embedding.
_BLOCKED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".ico",
        ".webp",
        ".svg",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".flac",
        ".ogg",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".onnx",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".pyc",
        ".pyo",
        ".DS_Store",
        ".lock",
    }
)


# ── Path safety helpers ───────────────────────────────────────────────────────


def _is_under_allowed_root(path: Path) -> bool:
    """Return True if *path* resolves to a location under an allowed root."""
    resolved = path.resolve()
    return any(str(resolved).startswith(str(root)) for root in _ALLOWED_INGEST_ROOTS)


def _safe_file_list(directory: Path) -> list[Path]:
    """
    Walk *directory* and return only files that are safe to ingest.

    Safety rules
    ------------
    * ``followlinks=False`` prevents symlink-based directory escape.
    * Each candidate file is resolved to its absolute real path and validated
      against ``_ALLOWED_INGEST_ROOTS`` before inclusion.
    * Hidden files (names starting with ``.``) are skipped.
    * Files whose extension is in ``_BLOCKED_EXTENSIONS`` are skipped.

    Parameters
    ----------
    directory:
        The directory to walk.  Must already be validated by the caller.

    Returns
    -------
    list[Path]
        Sorted list of ``Path`` objects that are safe and indexable.
    """
    valid_paths: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
        # Skip hidden directories in-place to prevent descending into them.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            # Skip hidden files.
            if filename.startswith("."):
                logger.debug("ingest_skip_hidden", file=filename)
                continue

            file_path = Path(dirpath) / filename

            # Skip blocked extensions.
            if file_path.suffix.lower() in _BLOCKED_EXTENSIONS:
                logger.debug(
                    "ingest_skip_blocked_ext",
                    file=str(file_path),
                    ext=file_path.suffix,
                )
                continue

            # Validate the resolved path is under an allowed root.
            try:
                resolved = file_path.resolve()
            except OSError as exc:
                logger.warning("ingest_resolve_error", file=str(file_path), error=str(exc))
                continue

            if not _is_under_allowed_root(resolved):
                logger.warning(
                    "ingest_path_traversal_blocked",
                    file=str(file_path),
                    resolved=str(resolved),
                )
                continue

            valid_paths.append(resolved)

    valid_paths.sort()
    logger.info(
        "ingest_file_list_built",
        directory=str(directory),
        file_count=len(valid_paths),
    )
    return valid_paths


# ── Public API ────────────────────────────────────────────────────────────────


def ingest_directory(
    directory: str | Path,
    collection: str = "knowledge_base",
) -> int:
    """
    Ingest all indexable documents in *directory* into the vector store.

    Pipeline
    --------
    1. Validate *directory* is under an allowed root.
    2. Build a safe file list via ``_safe_file_list``.
    3. Load documents with ``SimpleDirectoryReader``.
    4. Run ``IngestionPipeline``:
       - ``SentenceSplitter`` (chunk_size / overlap from ``cfg.ingestion``)
       - ``TitleExtractor`` (optional, controlled by ``cfg.ingestion.use_title_extractor``)
    5. Embed nodes using the HuggingFace model configured by ``setup_llamaindex``.
    6. Persist nodes to the vector store returned by ``get_vector_store(collection)``.

    Parameters
    ----------
    directory:
        Path to the directory containing source documents.  Must be a
        descendant of one of the ``_ALLOWED_INGEST_ROOTS``.
    collection:
        Target vector store collection name.  Defaults to ``"knowledge_base"``.

    Returns
    -------
    int
        Number of nodes (chunks) successfully indexed.

    Raises
    ------
    IngestionError
        On path-traversal attempts, missing directory, or pipeline failures.
    """
    try:
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex  # noqa: F401
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError as exc:
        raise IngestionError(
            "LlamaIndex is not installed. Run: pip install llama-index-core"
        ) from exc

    # ── 1. Validate and resolve directory ─────────────────────────────────────
    directory = Path(directory)

    if not directory.exists():
        raise IngestionError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise IngestionError(f"Path is not a directory: {directory}")

    resolved_dir = directory.resolve()
    if not _is_under_allowed_root(resolved_dir):
        raise IngestionError(
            f"Directory '{resolved_dir}' is outside the allowed ingestion roots: "
            f"{[str(r) for r in _ALLOWED_INGEST_ROOTS]}"
        )

    logger.info(
        "ingest_start",
        directory=str(resolved_dir),
        collection=collection,
    )

    # ── 2. Build safe file list ────────────────────────────────────────────────
    file_paths = _safe_file_list(resolved_dir)
    if not file_paths:
        logger.warning("ingest_no_files_found", directory=str(resolved_dir))
        return 0

    logger.info("ingest_files_discovered", count=len(file_paths))

    # ── 3. Load documents ─────────────────────────────────────────────────────
    try:
        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(
            input_files=[str(p) for p in file_paths],
        )
        documents = reader.load_data()
    except Exception as exc:
        raise IngestionError(f"SimpleDirectoryReader failed on '{resolved_dir}': {exc}") from exc

    logger.info("ingest_documents_loaded", document_count=len(documents))

    # ── 4. Build ingestion pipeline ───────────────────────────────────────────
    # Read chunk parameters from YAML config (with safe fallbacks).
    ingestion_cfg: dict = cfg._data.get("ingestion", {}) if hasattr(cfg, "_data") else {}
    chunk_size: int = int(ingestion_cfg.get("chunk_size", 512))
    chunk_overlap: int = int(ingestion_cfg.get("chunk_overlap", 64))
    use_title_extractor: bool = bool(ingestion_cfg.get("use_title_extractor", False))

    logger.info(
        "ingest_pipeline_config",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_title_extractor=use_title_extractor,
    )

    transformations: list = [
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    ]

    # TitleExtractor requires an LLM; only enable when quota is abundant.
    if use_title_extractor:
        try:
            from llama_index.core import Settings as LISettings
            from llama_index.core.extractors import TitleExtractor

            # Our architecture sets Settings.llm = None; TitleExtractor needs a
            # real LLM.  Warn rather than crash silently.
            if LISettings.llm is None:
                logger.warning(
                    "ingest_title_extractor_skipped",
                    reason=(
                        "Settings.llm is None — TitleExtractor requires an LLM. "
                        "Set use_title_extractor: false or configure Settings.llm."
                    ),
                )
            else:
                transformations.append(TitleExtractor())
                logger.info("ingest_title_extractor_enabled")
        except ImportError:
            logger.warning(
                "ingest_title_extractor_import_failed",
                reason="TitleExtractor not installed; skipping.",
            )

    # ── 5. Run pipeline and persist to vector store ───────────────────────────
    try:
        vector_store = get_vector_store(collection)
        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store,
        )
        nodes = pipeline.run(documents=documents, show_progress=False)
    except Exception as exc:
        raise IngestionError(
            f"IngestionPipeline failed for collection '{collection}': {exc}"
        ) from exc

    node_count = len(nodes)
    logger.info(
        "ingest_complete",
        collection=collection,
        nodes_indexed=node_count,
        documents_processed=len(documents),
    )
    return node_count


def load_index(collection: str = "knowledge_base") -> VectorStoreIndex:
    """
    Load an existing LlamaIndex ``VectorStoreIndex`` from the vector store.

    This is the read path used by retrieval tools at query time.
    ``setup_llamaindex()`` must be called before this function so that the
    global embed model is initialised.

    Parameters
    ----------
    collection:
        The collection name in the vector store to load.
        Defaults to ``"knowledge_base"``.

    Returns
    -------
    VectorStoreIndex
        A LlamaIndex index ready to create query engines against.

    Raises
    ------
    IngestionError
        If the vector store cannot be resolved or the index fails to load.
    """
    try:
        from llama_index.core import VectorStoreIndex
    except ImportError as exc:
        raise IngestionError(
            "LlamaIndex is not installed. Run: pip install llama-index-core"
        ) from exc

    logger.info("load_index_start", collection=collection)

    try:
        store = get_vector_store(collection)
        index = VectorStoreIndex.from_vector_store(store)
    except Exception as exc:
        raise IngestionError(f"Failed to load index for collection '{collection}': {exc}") from exc

    logger.info("load_index_complete", collection=collection)
    return index
