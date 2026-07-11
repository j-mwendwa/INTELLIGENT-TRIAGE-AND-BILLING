"""
src/core/logging.py
───────────────────
Structured-logging setup for the Intelligent Triage & Billing RAG system.

Usage
-----
Call ``setup_logging()`` once at application start-up (e.g. in ``main.py``
or the FastAPI ``lifespan`` handler).  Every module then gets a logger via::

    from src.core.logging import logger
    logger.info("event_name", key="value")

Request-scoped correlation IDs are stored in structlog context-vars so that
every log line emitted within a request automatically carries
``request_id=<uuid>``.  Bind / clear them with::

    from src.core.logging import bind_request_id, clear_request_context
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
)

# ──────────────────────────────────────────────────────────────────────────────
# Public convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

# Module-level logger – importable by all other modules.
# The actual processor chain is applied when setup_logging() is called.
logger: structlog.stdlib.BoundLogger = structlog.get_logger()


def bind_request_id(request_id: str) -> None:
    """
    Bind ``request_id`` to the current structlog context-var store.

    Call this at the start of every incoming request (e.g. FastAPI middleware).
    All subsequent ``logger.*`` calls in the same async-task context will
    automatically include ``request_id`` in their output.

    Parameters
    ----------
    request_id:
        A UUID string (or any opaque identifier) for the in-flight request.
    """
    bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    """
    Remove all context-var bindings for the current async task.

    Call this in a ``finally`` block or response middleware teardown to
    prevent stale values leaking into the next request that reuses the
    same worker.
    """
    clear_contextvars()


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────


def setup_logging(log_level: str = "INFO", app_env: str = "development") -> None:
    """
    Configure structlog and the stdlib root logger.

    Must be called **once** at application start-up before any logging occurs.

    Parameters
    ----------
    log_level:
        Standard log-level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, or ``"CRITICAL"``.  Case-insensitive.
    app_env:
        The deployment environment.  Any value other than ``"production"``
        (case-insensitive) activates the human-friendly ConsoleRenderer.
        In production, a JSON renderer is used for machine parsing.

    Behaviour
    ---------
    * **development / staging / test** → colourised, indented ``ConsoleRenderer``
      output suitable for reading in a terminal.
    * **production** → compact JSON output, one object per line, ready for
      ingestion by log aggregators (Cloud Logging, Datadog, etc.).

    Both modes:
    * Merge structlog context-vars (request_id, etc.) into every event dict.
    * Add log level, logger name, and a UTC ISO-8601 timestamp.
    * Render stdlib log records through the same processor chain.
    """
    numeric_level: int = getattr(logging, log_level.upper(), logging.INFO)
    is_production: bool = app_env.lower() == "production"

    # ── Shared processors (run before the final renderer) ─────────────────────
    shared_processors: list[Any] = [
        # Pull context-vars (request_id, etc.) into every event dict.
        merge_contextvars,
        # Add log level as a string field.
        structlog.stdlib.add_log_level,
        # Add the logger name (module path).
        structlog.stdlib.add_logger_name,
        # Render any positional args as the "event" field.
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Attach a UTC ISO-8601 timestamp.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Expand %s-style format strings left by stdlib logging.
        structlog.stdlib.ExtraAdder(),
        # Pretty-print exception tracebacks inside the event dict.
        structlog.processors.StackInfoRenderer(),
    ]

    if is_production:
        # ── Production: compact, machine-parseable JSON ────────────────────────
        final_renderer: Any = structlog.processors.JSONRenderer()
        structlog.configure(
            processors=shared_processors
            + [
                # Render exceptions as dicts (not plain strings) for JSON.
                structlog.processors.dict_tracebacks,
                final_renderer,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # ── Non-production: colourised human-readable output ───────────────────
        structlog.configure(
            processors=shared_processors
            + [
                # Format exceptions as a pretty indented string.
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    exception_formatter=structlog.dev.plain_traceback,
                ),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    # ── Route stdlib logging through structlog ─────────────────────────────────
    # This makes third-party libraries (httpx, uvicorn, etc.) appear in the
    # same structured format instead of raw stdlib output.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if is_production else sys.stderr,
        level=numeric_level,
    )
    # Replace the default handler with structlog's stdlib handler.
    handler = logging.StreamHandler(sys.stdout if is_production else sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer()
            if is_production
            else structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    _bootstrap_logger = structlog.get_logger("core.logging")
    _bootstrap_logger.info(
        "logging_configured",
        log_level=log_level.upper(),
        app_env=app_env,
        renderer="JSONRenderer" if is_production else "ConsoleRenderer",
    )
