"""
src/graph/guardrails.py — Input and output guardrail nodes.

Input guardrails:
  - NFKC + leet-speak normalisation
  - Length cap (cfg.security.max_input_length, default 4000 chars)
  - Blocked-phrase check
  - Injection-pattern regex

Output guardrails:
  - Empty-answer fallback
  - Runaway truncation (cfg.security.max_output_length, default 16000 chars)
"""

from __future__ import annotations

import re
import unicodedata

import structlog

from src.graph.state import AgentState

logger = structlog.get_logger(__name__)

# ── Leet-speak normalisation table ───────────────────────────────────────────
_LEET_TABLE = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "@": "a",
        "$": "s",
        "!": "i",
        "5": "s",
    }
)

# ── Injection patterns ────────────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(prior|previous)", re.I),
    re.compile(r"system\s+prompt\s+leak", re.I),
    re.compile(r"reveal\s+(your\s+)?(instructions|prompt|system)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?DAN", re.I),
    re.compile(r"pretend\s+you\s+(have\s+no|don.t\s+have)\s+restrictions", re.I),
    re.compile(r"act\s+as\s+(if\s+)?you\s+are\s+not\s+an?\s+AI", re.I),
]


def _normalise(text: str) -> str:
    """NFKC unicode normalisation + leet-speak substitution."""
    normalised = unicodedata.normalize("NFKC", text)
    return normalised.translate(_LEET_TABLE).lower()


def input_guard_node(state: AgentState) -> AgentState:
    """Evaluate user input against all guardrail checks."""
    from src.config import cfg

    task = state.get("task", "")

    # ── 1. Length cap ────────────────────────────────────────────────────────
    max_len: int = cfg._data.get("security", {}).get("max_input_length", 4000)
    if len(task) > max_len:
        logger.warning("input_too_long", length=len(task), max=max_len)
        return {
            **state,
            "input_security": {
                "decision": "BLOCK",
                "reason": f"Input exceeds maximum length of {max_len} characters.",
            },
        }

    normalised = _normalise(task)

    # ── 2. Blocked phrases ───────────────────────────────────────────────────
    blocked_phrases: list[str] = cfg._data.get("security", {}).get("blocked_phrases", [])
    for phrase in blocked_phrases:
        if phrase.lower() in normalised:
            logger.warning("blocked_phrase_detected", phrase=phrase)
            return {
                **state,
                "input_security": {"decision": "BLOCK", "reason": "Blocked content detected."},
            }

    # ── 3. Injection patterns ────────────────────────────────────────────────
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(normalised):
            logger.warning("injection_pattern_detected", pattern=pattern.pattern)
            return {
                **state,
                "input_security": {
                    "decision": "BLOCK",
                    "reason": "Potential prompt injection detected.",
                },
            }

    logger.debug("input_guard_passed", task_length=len(task))
    return {**state, "input_security": {"decision": "PASS", "reason": "OK"}}


def output_guard_node(state: AgentState) -> AgentState:
    """Apply output guardrails before returning to client."""
    from src.config import cfg

    answer = state.get("final_answer") or ""
    max_out: int = cfg._data.get("security", {}).get("max_output_length", 16000)

    # ── 1. Empty answer fallback ─────────────────────────────────────────────
    if not answer.strip():
        logger.warning("empty_answer_fallback")
        answer = (
            "I'm sorry, I wasn't able to generate a response to your query. "
            "Please try rephrasing your question or contact support directly."
        )

    # ── 2. Truncation cap ────────────────────────────────────────────────────
    if len(answer) > max_out:
        logger.warning("output_truncated", original_length=len(answer), cap=max_out)
        answer = answer[:max_out] + "\n\n[Response truncated due to length limit.]"

    return {**state, "final_answer": answer}


def rejection_node(state: AgentState) -> AgentState:
    """Return a safe rejection message for blocked inputs."""
    reason = (state.get("input_security") or {}).get("reason", "Input not allowed.")
    logger.info("input_rejected", reason=reason)
    return {
        **state,
        "final_answer": (
            f"Your request could not be processed: {reason} "
            "Please rephrase your question and try again."
        ),
    }
