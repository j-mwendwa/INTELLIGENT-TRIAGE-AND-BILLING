"""
src/core/tracing.py
───────────────────
LangSmith opt-in distributed tracing for the Intelligent Triage & Billing
RAG system.

LangSmith tracing works by setting a small set of environment variables that
the ``langchain-core`` runtime checks at import time (and again on each call).
This module centralises that configuration so no other file needs to touch
``os.environ`` directly.

Usage
-----
Call ``setup_langsmith()`` once during application start-up, ideally *before*
any LangChain / LangGraph objects are instantiated::

    from src.core.tracing import setup_langsmith
    setup_langsmith(api_key=settings.langsmith_api_key,
                    project=settings.langsmith_project)

Tracing is silently *disabled* when ``api_key`` is empty / None so that local
development environments that haven't configured LangSmith still work without
errors.
"""

from __future__ import annotations

import os

import structlog

logger = structlog.get_logger(__name__)

# Names of environment variables consumed by langchain-core / langsmith SDK.
_ENV_TRACING_V2 = "LANGCHAIN_TRACING_V2"
_ENV_API_KEY = "LANGCHAIN_API_KEY"
_ENV_PROJECT = "LANGCHAIN_PROJECT"
# Optional: endpoint override (useful for self-hosted LangSmith instances).
_ENV_ENDPOINT = "LANGCHAIN_ENDPOINT"

# Default public LangSmith endpoint.
_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


def setup_langsmith(
    api_key: str,
    project: str,
    *,
    endpoint: str = _DEFAULT_ENDPOINT,
    enabled: bool = True,
) -> bool:
    """
    Configure LangSmith distributed tracing via environment variables.

    Parameters
    ----------
    api_key:
        LangSmith API key (``ls__...``).  If empty or ``None``, tracing is
        automatically disabled regardless of the ``enabled`` flag.
    project:
        LangSmith project name.  All traces will be grouped under this
        project in the LangSmith UI.
    endpoint:
        LangSmith ingestion endpoint.  Defaults to the public cloud endpoint.
        Override for self-hosted / on-premise deployments.
    enabled:
        Explicit on/off switch.  Set to ``False`` to unconditionally disable
        tracing (e.g. in CI pipelines or when the feature flag is off).

    Returns
    -------
    bool
        ``True`` if tracing was successfully enabled, ``False`` otherwise.

    Side Effects
    ------------
    Sets (or unsets) the following environment variables in the current
    process so that ``langchain-core`` picks them up automatically:

    * ``LANGCHAIN_TRACING_V2``  → ``"true"`` / ``"false"``
    * ``LANGCHAIN_API_KEY``     → *api_key* (only when enabling)
    * ``LANGCHAIN_PROJECT``     → *project* (only when enabling)
    * ``LANGCHAIN_ENDPOINT``    → *endpoint* (only when enabling)
    """
    # Guard: disable if API key is absent or the caller explicitly opted out.
    if not enabled or not api_key:
        _disable_tracing(reason="no_api_key" if not api_key else "disabled_by_caller")
        return False

    # Set the variables that langchain-core / langsmith-sdk reads.
    os.environ[_ENV_TRACING_V2] = "true"
    os.environ[_ENV_API_KEY] = api_key
    os.environ[_ENV_PROJECT] = project
    os.environ[_ENV_ENDPOINT] = endpoint

    logger.info(
        "langsmith_tracing_enabled",
        project=project,
        endpoint=endpoint,
        # Redact the key — log only the first 8 chars so incidents can be
        # cross-referenced without exposing the full secret.
        api_key_prefix=api_key[:8] + "..." if len(api_key) > 8 else "***",
    )
    return True


def disable_tracing() -> None:
    """
    Explicitly disable LangSmith tracing and remove all related env vars.

    Useful in test fixtures or when toggling tracing at runtime.
    """
    _disable_tracing(reason="explicit_disable")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _disable_tracing(reason: str) -> None:
    """Set ``LANGCHAIN_TRACING_V2=false`` and scrub key / project vars."""
    os.environ[_ENV_TRACING_V2] = "false"
    # Remove sensitive vars so they don't accidentally leak into child
    # processes or crash the langsmith SDK when it tries to validate them.
    for var in (_ENV_API_KEY, _ENV_PROJECT, _ENV_ENDPOINT):
        os.environ.pop(var, None)

    logger.info(
        "langsmith_tracing_disabled",
        reason=reason,
    )


def is_tracing_enabled() -> bool:
    """
    Return ``True`` if LangSmith tracing is currently active in this process.

    Reads ``LANGCHAIN_TRACING_V2`` from the environment at call-time, so it
    reflects any changes made after ``setup_langsmith()`` was called.
    """
    return os.environ.get(_ENV_TRACING_V2, "false").lower() == "true"


def traceable(
    func=None,
    *,
    name: str | None = None,
    run_type: str = "chain",
    **kwargs,
):
    """
    Decorator that wraps a function with a LangSmith trace span.
    Falls back to a transparent no-op when tracing is disabled or LangSmith unavailable.

    Usage::

        @traceable(name="node.supervisor")
        def supervisor_node(state): ...
    """
    def decorator(fn):
        if not is_tracing_enabled():
            return fn
        try:
            from langsmith import traceable as _ls_traceable  # type: ignore
            span_name = name or fn.__qualname__
            return _ls_traceable(name=span_name, run_type=run_type, **kwargs)(fn)
        except Exception:
            return fn

    if func is not None:
        return decorator(func)
    return decorator
