"""
src/api/routes.py — All API route handlers.

Router prefix : /api/v1
Auth          : X-API-Key header (via require_api_key dependency)

Endpoints
---------
POST   /api/v1/chat                 — Single-turn chat (blocking)
POST   /api/v1/chat/stream          — Streaming chat (SSE)
DELETE /api/v1/threads/{thread_id}  — Delete thread entity memory
POST   /api/v1/ingest               — Ingest server-side path
POST   /api/v1/ingest/upload        — Upload & ingest multipart files
GET    /api/v1/health               — Liveness probe (no auth)
GET    /api/v1/version              — Version info (no auth)
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import aiofiles
import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.auth import require_api_key
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ThreadDeleteResponse,
)
from src.config import UPLOADS_DIR, settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["triage"])

# ── Allowed ingestion roots (server-side path validation) ─────────────────────
# Only paths under these directories may be submitted via POST /ingest.
# Adjust or extend as needed for your deployment.
_ALLOWED_INGEST_ROOTS: list[Path] = [
    Path("/home/jmwenda/AI/INTELLIGENT TRIAGE & BILLING/TRIAGE/data"),
    UPLOADS_DIR,
]


def _assert_path_allowed(raw_path: str) -> Path:
    """Resolve *raw_path* and verify it sits under an allowed root.

    Raises
    ------
    HTTPException(400)
        When the resolved path escapes all allowed root directories.
    """
    try:
        resolved = Path(raw_path).resolve(strict=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc

    for root in _ALLOWED_INGEST_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue

    raise HTTPException(
        status_code=400,
        detail=(
            f"Path '{raw_path}' is outside the allowed ingestion directories. "
            "Only paths under the configured data roots are permitted."
        ),
    )


# ── Helper: lazy import for ingestion ────────────────────────────────────────


def _get_ingest_directory():
    """Deferred import so ingestion pipeline does not slow startup."""
    try:
        from src.ingestion.pipeline import ingest_directory  # type: ignore[import]

        return ingest_directory
    except ImportError:
        # Fallback stub if ingestion pipeline is not yet implemented
        async def _stub(path: str, collection: str) -> int:  # noqa: RUF029
            raise HTTPException(
                status_code=501,
                detail="Ingestion pipeline not yet implemented.",
            )

        return _stub


# ── Chat — blocking ───────────────────────────────────────────────────────────


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Single-turn chat",
    description=(
        "Submit a user message and receive the full answer once the "
        "LangGraph pipeline has completed. Use ``/chat/stream`` for "
        "token-by-token SSE responses."
    ),
)
async def chat(
    req: ChatRequest,
    _: str = Depends(require_api_key),
) -> ChatResponse:
    """Handle a single blocking chat turn."""
    from src.graph.checkpointer import run_turn_stream

    thread_id = req.thread_id or str(uuid.uuid4())

    logger.info(
        "chat_request",
        thread_id=thread_id,
        message_len=len(req.message),
        entity_keys=list(req.entity_memory.keys()),
    )

    t_start = time.perf_counter()

    # Consume the async stream to collect the final "done" event
    final_event: dict = {}
    async for event in run_turn_stream(
        task=req.message,
        thread_id=thread_id,
        entity_memory=dict(req.entity_memory),
    ):
        if event.get("type") == "done":
            final_event = event.get("data", {})

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    answer = final_event.get("final_answer") or ""
    if not answer:
        logger.warning("chat_empty_answer", thread_id=thread_id)
        answer = "I was unable to produce a response. Please try again."

    logger.info(
        "chat_response",
        thread_id=thread_id,
        intent=final_event.get("intent"),
        active_subagent=final_event.get("active_subagent"),
        elapsed_ms=round(elapsed_ms, 2),
    )

    return ChatResponse(
        answer=answer,
        thread_id=thread_id,
        intent=final_event.get("intent"),
        active_subagent=final_event.get("active_subagent"),
        sources=final_event.get("sources") or [],
        routing_confidence=final_event.get("routing_confidence"),
        processing_time_ms=round(elapsed_ms, 2),
    )


# ── Chat — streaming (SSE) ────────────────────────────────────────────────────


@router.post(
    "/chat/stream",
    summary="Streaming chat (Server-Sent Events)",
    description=(
        "Submit a user message and receive incremental SSE events while the "
        "LangGraph pipeline runs. Each event is a JSON object with ``type`` "
        "and ``data`` fields. Event types: ``token``, ``node``, ``done``."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE stream",
            "content": {"text/event-stream": {}},
        }
    },
)
async def chat_stream(
    req: ChatRequest,
    _: str = Depends(require_api_key),
) -> StreamingResponse:
    """Return a StreamingResponse that yields SSE events from run_turn_stream."""
    from src.graph.checkpointer import run_turn_stream

    thread_id = req.thread_id or str(uuid.uuid4())

    logger.info(
        "chat_stream_request",
        thread_id=thread_id,
        message_len=len(req.message),
    )

    async def _event_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in run_turn_stream(
                task=req.message,
                thread_id=thread_id,
                entity_memory=dict(req.entity_memory),
            ):
                # SSE wire format: "data: <json>\n\n"
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.error("chat_stream_error", thread_id=thread_id, error=str(exc))
            error_event = {"type": "error", "data": {"message": str(exc)}}
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            # Signal stream end to any proxy / client
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Thread-ID": thread_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ── Thread management ─────────────────────────────────────────────────────────


@router.delete(
    "/threads/{thread_id}",
    response_model=ThreadDeleteResponse,
    summary="Delete thread entity memory",
    description=(
        "Permanently delete the persisted entity-memory file for the given "
        "thread. The thread's checkpoint history in SQLite is **not** removed "
        "by this endpoint."
    ),
)
async def delete_thread(
    thread_id: str,
    _: str = Depends(require_api_key),
) -> ThreadDeleteResponse:
    """Delete entity memory for a conversation thread."""
    from src.memory.entity_memory import EntityMemory

    logger.info("thread_delete_request", thread_id=thread_id)

    mem = EntityMemory(thread_id)
    mem_file = mem._path  # EntityMemory stores its path as ._path

    if not mem_file.exists():
        logger.warning("thread_delete_not_found", thread_id=thread_id)
        return ThreadDeleteResponse(thread_id=thread_id, status="not_found")

    mem.delete()
    logger.info("thread_delete_ok", thread_id=thread_id)
    return ThreadDeleteResponse(thread_id=thread_id, status="deleted")


# ── Ingestion — server-side path ──────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a server-side file or directory",
    description=(
        "Trigger ingestion of a file or directory that already exists on the "
        "server. The ``path`` must be under one of the allowed data roots. "
        "Use ``/ingest/upload`` to push documents from the client."
    ),
)
async def ingest(
    req: IngestRequest,
    _: str = Depends(require_api_key),
) -> IngestResponse:
    """Ingest a server-side path into the vector store."""
    resolved_path = _assert_path_allowed(req.path)

    if not resolved_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Path does not exist on the server: {resolved_path}",
        )

    logger.info(
        "ingest_request",
        path=str(resolved_path),
        collection=req.collection,
    )

    ingest_directory = _get_ingest_directory()
    try:
        docs_indexed: int = await ingest_directory(
            path=str(resolved_path),
            collection=req.collection,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ingest_error", path=str(resolved_path), error=str(exc))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    logger.info(
        "ingest_complete",
        path=str(resolved_path),
        collection=req.collection,
        docs_indexed=docs_indexed,
    )

    return IngestResponse(
        status="ok",
        collection=req.collection,
        documents_indexed=docs_indexed,
    )


# ── Ingestion — file upload ───────────────────────────────────────────────────


@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    summary="Upload files and ingest",
    description=(
        "Upload one or more files via multipart form-data. Files are saved "
        "to ``data/uploads/`` and then ingested into the specified collection."
    ),
)
async def ingest_upload(
    files: list[UploadFile],
    collection: str = "knowledge_base",
    _: str = Depends(require_api_key),
) -> IngestResponse:
    """Accept uploaded files, save them, and kick off ingestion."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="All files must have a filename.")

        # Sanitise filename — strip directory components
        safe_name = Path(upload.filename).name
        dest = UPLOADS_DIR / safe_name

        logger.info("upload_saving", filename=safe_name, destination=str(dest))

        try:
            async with aiofiles.open(dest, "wb") as fh:
                while chunk := await upload.read(1024 * 64):  # 64 KiB chunks
                    await fh.write(chunk)
        except Exception as exc:
            logger.error("upload_save_error", filename=safe_name, error=str(exc))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save '{safe_name}': {exc}",
            ) from exc
        finally:
            await upload.close()

        saved_paths.append(dest)

    logger.info(
        "upload_complete",
        file_count=len(saved_paths),
        collection=collection,
    )

    # Ingest the uploads directory (or individual files)
    ingest_directory = _get_ingest_directory()
    total_indexed = 0
    errors: list[str] = []

    for path in saved_paths:
        try:
            count: int = await ingest_directory(
                path=str(path),
                collection=collection,
            )
            total_indexed += count
        except Exception as exc:
            logger.error("upload_ingest_error", path=str(path), error=str(exc))
            errors.append(f"{path.name}: {exc}")

    if errors and not total_indexed:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed for all files: {'; '.join(errors)}",
        )

    return IngestResponse(
        status="ok" if not errors else "partial",
        collection=collection,
        documents_indexed=total_indexed,
    )


# ── Health & Version (no auth) ────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns HTTP 200 when the service is healthy. No authentication required.",
    tags=["ops"],
)
async def health() -> HealthResponse:
    """Liveness probe — always returns 200 when the process is alive."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.app_env,
    )


@router.get(
    "/version",
    summary="Version info",
    description="Returns the service name and version. No authentication required.",
    tags=["ops"],
)
async def version() -> dict:
    """Return service name and version."""
    return {
        "version": "0.1.0",
        "service": "Intelligent Triage & Billing",
        "environment": settings.app_env,
    }
