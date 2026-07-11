"""
src/api/schemas.py — Pydantic v2 request / response models for the
Intelligent Triage & Billing API.

All models use ``model_config = ConfigDict(str_strip_whitespace=True)``
so that accidental leading/trailing whitespace in user-supplied strings is
silently stripped before validation.

Models
------
ChatRequest         POST /api/v1/chat
ChatResponse        POST /api/v1/chat  (response)
IngestRequest       POST /api/v1/ingest
IngestResponse      POST /api/v1/ingest  (response)
HealthResponse      GET  /api/v1/health
ThreadDeleteResponse DELETE /api/v1/threads/{thread_id}
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ── Chat ─────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat turn payload."""

    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's message or query (1–4 000 characters).",
        examples=["What does item 42 on my bill mean?"],
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation thread identifier. When ``null`` the API will "
            "generate a new UUID and return it in the response. Reuse the "
            "same thread_id across turns to maintain conversation history."
        ),
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    entity_memory: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional caller-supplied entity facts that are merged with any "
            "persisted entity memory for this thread before the turn runs. "
            "Keys and values must both be strings."
        ),
        examples=[{"patient_name": "Alice Wanjiru", "insurance_id": "NHIF-12345"}],
    )


class ChatResponse(BaseModel):
    """Response payload for a completed (non-streaming) chat turn."""

    answer: str = Field(
        ...,
        description="The final generated answer from the active subagent.",
    )
    thread_id: str = Field(
        ...,
        description="The thread identifier (echoed back or newly generated).",
    )
    intent: str | None = Field(
        default=None,
        description="Detected intent label, e.g. 'billing', 'technical', 'compliance'.",
    )
    active_subagent: str | None = Field(
        default=None,
        description="Name of the domain subagent that produced the answer.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Cited source document titles or URIs used to construct the answer.",
    )
    routing_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Supervisor routing confidence score in [0, 1].",
    )
    processing_time_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Wall-clock time in milliseconds to produce this response.",
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """Request body for server-side path ingestion."""

    model_config = ConfigDict(str_strip_whitespace=True)

    path: str = Field(
        ...,
        description=(
            "Absolute filesystem path (on the server) to a file or directory "
            "to ingest. The server validates that the path is under an "
            "allowed root before processing."
        ),
        examples=["/home/jmwenda/AI/INTELLIGENT TRIAGE & BILLING/TRIAGE/data/docs"],
    )
    collection: str = Field(
        default="knowledge_base",
        description="Target vector-store collection name.",
        examples=["knowledge_base", "billing_docs", "compliance_docs"],
    )


class IngestResponse(BaseModel):
    """Response payload after an ingestion job completes."""

    status: str = Field(
        ...,
        description="Final job status: 'ok' on success, 'error' on failure.",
    )
    collection: str = Field(
        ...,
        description="Name of the collection that was populated.",
    )
    documents_indexed: int = Field(
        ...,
        ge=0,
        description="Number of document chunks successfully indexed.",
    )


# ── Health & Version ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""

    status: str = Field(
        default="ok",
        description="Service health status. 'ok' when all subsystems are healthy.",
    )
    version: str = Field(
        ...,
        description="Deployed application version string (semver).",
    )
    environment: str = Field(
        ...,
        description="Runtime environment label, e.g. 'development' or 'production'.",
    )


# ── Thread management ─────────────────────────────────────────────────────────


class ThreadDeleteResponse(BaseModel):
    """Response payload after a thread's entity memory is deleted."""

    thread_id: str = Field(
        ...,
        description="The thread identifier that was deleted.",
    )
    status: str = Field(
        ...,
        description="Deletion outcome: 'deleted' or 'not_found'.",
    )
