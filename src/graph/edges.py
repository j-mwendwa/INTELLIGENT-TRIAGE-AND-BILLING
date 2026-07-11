"""
src/graph/edges.py — Edge routing functions for the LangGraph supervisor architecture.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.graph.state import AgentState


def route_after_input_check(state: AgentState) -> str:
    """After input guardrail: route to supervisor or rejection node."""
    security = state.get("input_security") or {}
    if security.get("decision") == "BLOCK":
        return "rejection"
    return "supervisor"


def route_supervisor(state: AgentState) -> str:
    """After supervisor: route to the appropriate domain subgraph."""
    intent = state.get("intent") or "general"
    routing_map = {
        "billing": "billing_agent",
        "technical": "technical_agent",
        "compliance": "compliance_agent",
        "escalate": "escalation",
        "general": "general_agent",
    }
    return routing_map.get(intent, "general_agent")


def should_continue(state: AgentState, agent_name: str) -> str:
    """After an agent node: decide whether to call tools or extract answer."""
    messages = state.get("messages", [])
    if not messages:
        return "extract_answer"

    last_message = messages[-1]
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 8)

    # Hit iteration limit
    if iteration >= max_iterations:
        return "extract_answer"

    # Has tool calls → go to tool node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # No tool calls → extract final answer
    return "extract_answer"


def billing_should_continue(state: AgentState) -> str:
    return should_continue(state, "billing")


def technical_should_continue(state: AgentState) -> str:
    return should_continue(state, "technical")


def compliance_should_continue(state: AgentState) -> str:
    return should_continue(state, "compliance")


def general_should_continue(state: AgentState) -> str:
    return should_continue(state, "general")


def route_after_tools(state: AgentState) -> str:
    """After tool execution: always return to the active agent."""
    intent = state.get("intent") or "general"
    routing_map = {
        "billing": "billing_agent",
        "technical": "technical_agent",
        "compliance": "compliance_agent",
        "general": "general_agent",
    }
    return routing_map.get(intent, "general_agent")
