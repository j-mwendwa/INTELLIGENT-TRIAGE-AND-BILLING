"""
src/graph/state.py — AgentState TypedDict that flows through every LangGraph node.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state for the supervisor and all subgraph agents."""

    # ── Conversation ─────────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    """Full conversation history. `add_messages` reducer appends new messages."""

    task: str
    """The current user input / turn."""

    # ── Memory ───────────────────────────────────────────────────────────────
    entity_memory: dict[str, str]
    """Long-term KV facts for this thread (persisted to disk)."""

    conversation_summary: str | None
    """Rolling Gemini-generated summary of the conversation."""

    # ── Retrieval ─────────────────────────────────────────────────────────────
    sources: list[str]
    """Knowledge base search queries used in this turn (citation trail)."""

    # ── Supervisor routing ────────────────────────────────────────────────────
    intent: str | None
    """Classified intent: 'billing' | 'technical' | 'compliance' | 'general' | 'escalate'."""

    routing_confidence: float | None
    """Supervisor confidence score for the routing decision (0.0 – 1.0)."""

    active_subagent: str | None
    """Name of the currently active subgraph: 'billing' | 'technical' | 'compliance'."""

    # ── Security ──────────────────────────────────────────────────────────────
    input_security: dict[str, Any] | None
    """Guardrail decision: {'decision': 'PASS'|'BLOCK', 'reason': str}."""

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: str | None
    """Synthesised answer to return to the client."""

    # ── Loop control ──────────────────────────────────────────────────────────
    iteration: int
    """Tool-call loop counter for the active subgraph."""

    max_iterations: int
    """Hard cap: stops infinite tool loops."""
