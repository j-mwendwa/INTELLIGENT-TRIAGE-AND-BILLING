"""
src/core/exceptions.py
─────────────────────
Custom exception hierarchy for the Intelligent Triage & Billing RAG system.

All application-level exceptions derive from TriageAgentError so callers
can catch the entire family with a single except clause when needed.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────


class TriageAgentError(Exception):
    """
    Root exception for every domain-specific error raised by the Triage agent.

    All sub-exceptions inherit from this class so that callers can write:
        except TriageAgentError as exc: ...
    to handle any application-level error in one place.
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:  # pragma: no cover
        return self.message or self.__class__.__name__


# ──────────────────────────────────────────────────────────────────────────────
# Sub-exceptions
# ──────────────────────────────────────────────────────────────────────────────


class IngestionError(TriageAgentError):
    """
    Raised when document ingestion into the vector store fails.

    Examples
    --------
    - File cannot be parsed (corrupt PDF, unsupported format)
    - Embedding API returns an error mid-batch
    - Chunk size / overlap misconfiguration
    """


class ConfigError(TriageAgentError):
    """
    Raised for invalid, missing, or incompatible configuration values.

    Examples
    --------
    - Required environment variable is absent
    - Numeric parameter is out of range
    - Incompatible combination of settings (e.g., conflicting LLM options)
    """


class GuardrailError(TriageAgentError):
    """
    Raised when a safety or policy guardrail blocks an action.

    Attributes
    ----------
    reason : str
        Human-readable explanation of *why* the guardrail triggered.
        This value is safe to surface to the user or log to an audit trail.

    Examples
    --------
    - Prompt injection attempt detected
    - Output violates content-safety policy
    - Request exceeds authorised scope for the current user role
    """

    def __init__(self, message: str = "", *, reason: str = "") -> None:
        super().__init__(message)
        # ``reason`` carries the policy/rule name that was violated so that
        # monitoring dashboards can group guardrail violations by category.
        self.reason: str = reason

    def __str__(self) -> str:  # pragma: no cover
        parts = [self.message] if self.message else []
        if self.reason:
            parts.append(f"[reason={self.reason}]")
        return " ".join(parts) or self.__class__.__name__


class VectorStoreError(TriageAgentError):
    """
    Raised for failures related to the vector store (Qdrant / Chroma / etc.).

    Examples
    --------
    - Collection does not exist and auto-create is disabled
    - Network timeout when connecting to the vector DB
    - Incompatible index parameters on re-initialisation
    """


class MemoryError(TriageAgentError):
    """
    Raised when the conversation-memory subsystem encounters an error.

    Note: this intentionally shadows the built-in ``MemoryError`` within this
    package.  Import from ``src.core.exceptions`` explicitly to avoid
    ambiguity.

    Examples
    --------
    - Redis connection failure when persisting entity memory
    - Deserialization error on a stored memory snapshot
    - Memory budget exceeded and eviction policy is set to 'raise'
    """
