"""
src/api/main.py — FastAPI application factory.

Lifespan startup order:
  1. setup_logging
  2. setup_langsmith (opt-in)
  3. setup_llamaindex
  4. register_base_tools
  5. load_mcp_tools → register_mcp_tools
  6. get_app_async (compile LangGraph)
  7. Mount web/ as static files

Middleware stack (outer → inner):
  CORSMiddleware
  RequestIDMiddleware   → X-Request-ID
  TimingMiddleware      → X-Process-Time
  SecurityHeadersMiddleware
  SlowAPIMiddleware     → rate limiting
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import BASE_DIR, cfg, settings

logger = structlog.get_logger(__name__)

# ── Rate limiter (slowapi) ────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Middleware definitions ────────────────────────────────────────────────────


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject X-Request-ID header and bind to structlog context."""

    async def dispatch(self, request: Request, call_next):
        import structlog.contextvars as ctx

        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        ctx.clear_contextvars()
        ctx.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Add X-Process-Time header (wall-clock ms)."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security response headers."""

    _HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Cache-Control": "no-store",
    }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for key, value in self._HEADERS.items():
            response.headers[key] = value
        return response


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown sequence."""
    # 1. Logging
    from src.core.logging import setup_logging

    setup_logging(log_level=settings.log_level, app_env=settings.app_env)
    logger.info("startup_begin", env=settings.app_env)

    # 2. LangSmith tracing (opt-in)
    from src.core.tracing import setup_langsmith

    setup_langsmith(api_key=settings.langsmith_api_key, project=settings.langsmith_project)

    # 3. LlamaIndex embeddings
    from src.core.llamaindex_setup import setup_llamaindex

    setup_llamaindex()

    # 4. Register base tools
    from src.tools.registry import register_base_tools

    register_base_tools()

    # 5. Load MCP tools
    from src.tools.mcp_client import load_mcp_tools
    from src.tools.registry import register_mcp_tools

    mcp_tools = await load_mcp_tools()
    register_mcp_tools(mcp_tools)

    # 6. Compile LangGraph (async singleton)
    from src.graph.graph import get_app_async

    await get_app_async()

    logger.info("startup_complete", mcp_tool_count=len(mcp_tools))

    yield  # ← app is running

    logger.info("shutdown_complete")


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    cfg._data.get("security", {}).get("rate_limit", "30/minute")

    app = FastAPI(
        title="Intelligent Triage & Billing API",
        description=(
            "Hierarchical multi-agent RAG system for intelligent support triage. "
            "Supervisor routes queries to Billing, Technical, or Compliance subgraphs."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Rate limiter state ────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Middleware (added in reverse — last added = outermost) ────────────────
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    # Non-production: allow all origins.
    # Production: use CORS_ORIGINS env var (comma-separated list of allowed origins).
    #   Example: https://my-app.azurecontainerapps.io,https://my-custom-domain.com
    if settings.app_env == "production":
        _cors_origins = settings.cors_origins or []
        if not _cors_origins:
            logger.warning(
                "cors_origins_empty",
                hint="Set CORS_ORIGINS env var to allow browser clients in production.",
            )
    else:
        _cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(
            "http_error", status=exc.status_code, detail=exc.detail, path=str(request.url)
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("validation_error", errors=exc.errors(), path=str(request.url))
        return JSONResponse(
            status_code=422,
            content={"error": "Validation error", "details": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_exception", error=str(exc), path=str(request.url), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    from src.api.routes import router

    app.include_router(router)

    # ── Static files (chat UI) ────────────────────────────────────────────────
    web_dir = BASE_DIR / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
        logger.info("static_files_mounted", path=str(web_dir))

    return app


# ── Module-level app instance (for uvicorn) ───────────────────────────────────
app = create_app()
