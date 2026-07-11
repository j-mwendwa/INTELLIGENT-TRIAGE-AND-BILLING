"""
src/core/prompt_manager.py
──────────────────────────
Prompt-file loader for the Intelligent Triage & Billing RAG system.

Prompt templates are stored as Markdown files under the ``prompts/`` directory
tree.  This module provides a single ``load_prompt()`` function that:

1. Searches several conventional paths for the requested prompt.
2. Caches loaded content in a module-level dict to avoid redundant I/O.
3. Logs cache hits and misses via structlog.
4. Raises ``FileNotFoundError`` with a clear, actionable message when no
   matching file is found.

Directory conventions
---------------------
Given ``name="triage_system"`` and ``version="v1"``, the loader searches (in
order) for::

    prompts/triage_system_v1.md
    prompts/triage_system/triage_system_v1.md
    prompts/system/triage_system_v1.md

The **project root** is resolved relative to this file's location, so the
paths always point to the correct location regardless of the working directory.

Usage
-----
    from src.core.prompt_manager import load_prompt

    system_prompt = load_prompt("triage_system")          # default version v1
    older_prompt  = load_prompt("triage_system", "v2")    # explicit version
"""

from __future__ import annotations

import pathlib
from typing import Final

import structlog

logger = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level prompt cache
# ──────────────────────────────────────────────────────────────────────────────

# Keys are ``"{name}:{version}"`` strings; values are the raw file contents.
_PROMPT_CACHE: dict[str, str] = {}

# ──────────────────────────────────────────────────────────────────────────────
# Resolve the project root and prompts directory
# ──────────────────────────────────────────────────────────────────────────────

# This file lives at  src/core/prompt_manager.py
# Project root is     ../..  from here  →  BASE_DIR
_THIS_FILE: Final[pathlib.Path] = pathlib.Path(__file__).resolve()
_PROJECT_ROOT: Final[pathlib.Path] = _THIS_FILE.parent.parent.parent
_PROMPTS_ROOT: Final[pathlib.Path] = _PROJECT_ROOT / "prompts"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def load_prompt(name: str, version: str = "v1") -> str:
    """
    Load and cache a prompt template from the ``prompts/`` directory tree.

    The function searches for ``{name}_{version}.md`` in the following
    locations (tried in order, first match wins):

    1. ``prompts/{name}_{version}.md``          — flat layout
    2. ``prompts/{name}/{name}_{version}.md``   — grouped-by-name layout
    3. ``prompts/system/{name}_{version}.md``   — system-prompt layout

    Parameters
    ----------
    name:
        Logical name of the prompt, without version suffix or file extension.
        Examples: ``"triage_system"``, ``"billing_classifier"``.
    version:
        Version tag appended to the filename.  Defaults to ``"v1"``.
        Examples: ``"v1"``, ``"v2"``, ``"prod"``.

    Returns
    -------
    str
        The full text content of the prompt file, stripped of leading/trailing
        whitespace.

    Raises
    ------
    FileNotFoundError
        If no matching file is found in any of the search paths.  The error
        message lists all paths that were tried so the developer knows exactly
        where to create the file.

    Examples
    --------
    >>> prompt = load_prompt("triage_system")
    >>> prompt = load_prompt("billing_classifier", version="v2")
    """
    cache_key = f"{name}:{version}"

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_key in _PROMPT_CACHE:
        logger.debug(
            "prompt_cache_hit",
            name=name,
            version=version,
            cache_key=cache_key,
        )
        return _PROMPT_CACHE[cache_key]

    # ── Search candidate paths ────────────────────────────────────────────────
    filename = f"{name}_{version}.md"
    candidates: list[pathlib.Path] = [
        _PROMPTS_ROOT / filename,
        _PROMPTS_ROOT / name / filename,
        _PROMPTS_ROOT / "system" / filename,
    ]

    logger.debug(
        "prompt_cache_miss",
        name=name,
        version=version,
        candidates=[str(p) for p in candidates],
    )

    for path in candidates:
        if path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            _PROMPT_CACHE[cache_key] = content
            logger.info(
                "prompt_loaded",
                name=name,
                version=version,
                path=str(path),
                chars=len(content),
            )
            return content

    # ── Not found ─────────────────────────────────────────────────────────────
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Prompt '{name}' (version '{version}') not found.\n"
        f"Searched the following paths (none exist):\n  {searched}\n\n"
        f"Create one of the files above with your prompt content to fix this."
    )


def clear_prompt_cache() -> None:
    """
    Evict all entries from the in-memory prompt cache.

    Useful in test suites that need to verify cache-miss / reload behaviour
    without restarting the process.
    """
    count = len(_PROMPT_CACHE)
    _PROMPT_CACHE.clear()
    logger.debug("prompt_cache_cleared", evicted=count)


def list_cached_prompts() -> list[str]:
    """
    Return the cache keys for all currently cached prompts.

    Cache keys have the format ``"{name}:{version}"``.
    """
    return list(_PROMPT_CACHE.keys())
