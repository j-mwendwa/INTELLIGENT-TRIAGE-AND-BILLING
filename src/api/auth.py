"""
src/api/auth.py — API Key authentication dependency.

Usage
-----
Protect any route by declaring the dependency::

    from src.api.auth import require_api_key

    @router.post("/chat")
    async def chat(req: ChatRequest, _: str = Depends(require_api_key)):
        ...

Security design
---------------
* The raw API key is **never** logged or stored beyond the in-memory
  ``settings.allowed_api_keys`` list.
* Every auth attempt (success or failure) logs the first 12 hex chars of
  the SHA-256 digest so that auditors can correlate events to a specific
  key without ever recovering the original value.
* A constant-time comparison (via ``hmac.compare_digest``) prevents
  timing-based enumeration of valid keys.
"""

from __future__ import annotations

import hashlib
import hmac

import structlog
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from src.config import settings

logger = structlog.get_logger(__name__)

# ── Header extractor ─────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,  # We raise our own exception for better logging
    description="API key required for all protected endpoints.",
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _key_fingerprint(key: str) -> str:
    """Return the first 12 hex chars of the SHA-256 digest of *key*.

    This is safe to log: it uniquely identifies a key without exposing it.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return digest[:12]


def _is_valid_key(candidate: str) -> bool:
    """Constant-time check of *candidate* against every allowed API key.

    Iterates all keys (not short-circuiting) to prevent timing attacks that
    could reveal how many keys are configured or which prefix matches.
    """
    result = False
    for allowed in settings.allowed_api_keys:
        # hmac.compare_digest requires same-type arguments
        if hmac.compare_digest(
            candidate.encode("utf-8"),
            allowed.encode("utf-8"),
        ):
            result = True
    return result


# ── FastAPI dependency ────────────────────────────────────────────────────────


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """FastAPI dependency — validates the ``X-API-Key`` header.

    Parameters
    ----------
    api_key:
        Extracted automatically from the ``X-API-Key`` header by
        :class:`~fastapi.security.APIKeyHeader`.

    Returns
    -------
    str
        The validated API key string (callers rarely need this, but
        returning it allows further inspection if required).

    Raises
    ------
    HTTPException(403)
        When the header is missing or the key is not in
        ``settings.allowed_api_keys``.
    """
    if api_key is None:
        logger.warning(
            "api_auth_missing_key",
            detail="X-API-Key header not provided",
        )
        raise HTTPException(
            status_code=403,
            detail="Missing X-API-Key header.",
        )

    fingerprint = _key_fingerprint(api_key)

    if not _is_valid_key(api_key):
        logger.warning(
            "api_auth_invalid_key",
            key_fingerprint=fingerprint,
            detail="Supplied key does not match any allowed key",
        )
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    logger.info(
        "api_auth_success",
        key_fingerprint=fingerprint,
    )
    return api_key
