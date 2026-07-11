"""
src/graph/graph.py — LangGraph multi-agent supervisor graph.

Architecture:
  input_guard → supervisor → [billing_agent | technical_agent | compliance_agent | general_agent | escalation]
                             ↓ (tool calls)
                          tool_node → back to active agent
                             ↓ (final answer)
                          extract_answer → output_guard → END
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import structlog
from langgraph.graph import END, StateGraph

from src.graph.state import AgentState

logger = structlog.get_logger(__name__)

# Singleton async app
_async_app = None
_async_app_lock = asyncio.Lock()


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph StateGraph."""
    from src.config import cfg
    from src.graph.edges import (
        billing_should_continue,
        compliance_should_continue,
        general_should_continue,
        route_after_input_check,
        route_after_tools,
        route_supervisor,
        technical_should_continue,
    )
    from src.graph.guardrails import input_guard_node, output_guard_node, rejection_node
    from src.graph.nodes import (
        billing_agent_node,
        compliance_agent_node,
        escalation_node,
        extract_final_answer_node,
        general_agent_node,
        supervisor_node,
        technical_agent_node,
        tool_node,
    )

    cfg._data.get("graph", {}).get("max_iterations", 8)

    # ── Build graph ──────────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────────────────────────
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("rejection", rejection_node)
    workflow.add_node("billing_agent", billing_agent_node)
    workflow.add_node("technical_agent", technical_agent_node)
    workflow.add_node("compliance_agent", compliance_agent_node)
    workflow.add_node("general_agent", general_agent_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("extract_answer", extract_final_answer_node)
    workflow.add_node("output_guard", output_guard_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    workflow.set_entry_point("input_guard")

    # ── Edges from input guard ────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "input_guard",
        route_after_input_check,
        {"rejection": "rejection", "supervisor": "supervisor"},
    )

    # ── Rejection → output guard ──────────────────────────────────────────────
    workflow.add_edge("rejection", "output_guard")

    # ── Supervisor → domain agents ────────────────────────────────────────────
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "billing_agent": "billing_agent",
            "technical_agent": "technical_agent",
            "compliance_agent": "compliance_agent",
            "general_agent": "general_agent",
            "escalation": "escalation",
        },
    )

    # ── Escalation → output guard ─────────────────────────────────────────────
    workflow.add_edge("escalation", "output_guard")

    # ── Domain agents → tools or extract_answer ────────────────────────────────
    for agent_name, should_continue_fn in [
        ("billing_agent", billing_should_continue),
        ("technical_agent", technical_should_continue),
        ("compliance_agent", compliance_should_continue),
        ("general_agent", general_should_continue),
    ]:
        workflow.add_conditional_edges(
            agent_name,
            should_continue_fn,
            {"tools": "tools", "extract_answer": "extract_answer"},
        )

    # ── Tools → back to active agent ──────────────────────────────────────────
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "billing_agent": "billing_agent",
            "technical_agent": "technical_agent",
            "compliance_agent": "compliance_agent",
            "general_agent": "general_agent",
        },
    )

    # ── Extract answer → output guard → END ───────────────────────────────────
    workflow.add_edge("extract_answer", "output_guard")
    workflow.add_edge("output_guard", END)

    return workflow


@lru_cache(maxsize=1)
def get_app():
    """Compile and return the sync LangGraph app (for tests/scripts)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    from src.config import DATA_DIR

    db_path = str(DATA_DIR / "checkpoints.db")
    checkpointer = SqliteSaver.from_conn_string(db_path)
    graph = build_graph()
    app = graph.compile(checkpointer=checkpointer)
    logger.info("langgraph_compiled_sync", db=db_path)
    return app


async def get_app_async():
    """Compile and return the async LangGraph app (FastAPI path, singleton)."""
    global _async_app
    async with _async_app_lock:
        if _async_app is None:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            from src.config import DATA_DIR

            db_path = str(DATA_DIR / "checkpoints.db")
            checkpointer = await AsyncSqliteSaver.from_conn_string(db_path)
            graph = build_graph()
            _async_app = graph.compile(checkpointer=checkpointer)
            logger.info("langgraph_compiled_async", db=db_path)
    return _async_app
