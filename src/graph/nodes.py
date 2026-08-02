"""
src/graph/nodes.py — All LangGraph node functions.

Hierarchy:
  supervisor_node  — classifies intent and routes
  agent_node       — generic agent that uses tools (used by subgraphs)
  tool_node        — dispatches tool calls and collects sources
  rejection_node   — called when input guardrail blocks (defined in guardrails.py)
"""

from __future__ import annotations

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.graph.state import AgentState
from src.core.tracing import traceable

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
#  LLM factory helpers
# ────────────────────────────────────────────────────────────────────────────


def _get_llm(model_key: str = "default_model"):
    """Return a ChatGoogleGenerativeAI instance for the specified model config key."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.config import cfg, settings

    llm_cfg = cfg._data.get("llm", {})
    model = llm_cfg.get(model_key, "gemini-2.5-flash")
    temperature = llm_cfg.get("temperature", 0.0)
    max_tokens = llm_cfg.get("max_tokens", 4096)

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=settings.google_api_key,
    )


# ────────────────────────────────────────────────────────────────────────────
#  Supervisor node
# ────────────────────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are the Intelligent Triage Supervisor for a support platform.

Your job is to classify the user's request into exactly one of these intents:
- billing    : questions about invoices, payments, refunds, pricing, account credits
- technical  : questions about service issues, errors, outages, API, features, SLAs
- compliance : questions about data privacy, regulations, legal, policies, security certs
- general    : questions that don't clearly fit the above three
- escalate   : urgent/emergency situations, legal threats, or multiple unresolvable issues

Respond in this exact JSON format:
{
  "intent": "<billing|technical|compliance|general|escalate>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>"
}
"""


@traceable(name="node.supervisor", run_type="llm")
def supervisor_node(state: AgentState) -> AgentState:
    """Classify the user's intent and set routing metadata."""
    import json

    task = state.get("task", "")
    entity_memory = state.get("entity_memory", {})
    conversation_summary = state.get("conversation_summary")

    # Build context for the supervisor
    context_parts = [SUPERVISOR_SYSTEM]
    if entity_memory:
        facts = "\n".join(f"- {k}: {v}" for k, v in entity_memory.items())
        context_parts.append(f"\n<known_facts>\n{facts}\n</known_facts>")
    if conversation_summary:
        context_parts.append(f"\n<summary>\n{conversation_summary}\n</summary>")

    system_content = "\n".join(context_parts)

    llm = _get_llm("default_model")
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Classify this request: {task}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        intent = parsed.get("intent", "general")
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "")

        logger.info(
            "supervisor_routed",
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            task=task[:80],
        )

        return {
            **state,
            "intent": intent,
            "routing_confidence": confidence,
            "active_subagent": intent if intent != "escalate" else None,
        }

    except Exception as exc:
        logger.warning("supervisor_parse_error", error=str(exc))
        return {
            **state,
            "intent": "general",
            "routing_confidence": 0.0,
            "active_subagent": "general",
        }


# ────────────────────────────────────────────────────────────────────────────
#  Agent node (generic — used by all domain subgraphs)
# ────────────────────────────────────────────────────────────────────────────


def _build_agent_node(
    agent_name: str,
    model_key: str,
    system_prompt_name: str,
    domain_tools: list[str],
):
    """Factory that creates a domain-specific agent_node closure."""

    @traceable(name="node.agent")
    def agent_node(state: AgentState) -> AgentState:
        from src.config import cfg
        from src.core.context_assembler import ContextAssembler
        from src.core.prompt_manager import load_prompt
        from src.core.token_counter import TokenCounter
        from src.tools.registry import get_tools

        task = state.get("task", "")
        messages = state.get("messages", [])
        entity_memory = state.get("entity_memory", {})
        conversation_summary = state.get("conversation_summary")

        # Load system prompt
        try:
            base_system = load_prompt(system_prompt_name)
        except FileNotFoundError:
            base_system = f"You are the {agent_name} support agent. Help users with their {agent_name}-related queries professionally and accurately."

        # Assemble context
        assembler = ContextAssembler(token_counter=TokenCounter())
        target_tokens = cfg._data.get("context", {}).get("target_context_tokens", 8000)
        system_content = assembler.build(
            system_prompt=base_system,
            entity_memory=entity_memory,
            conversation_summary=conversation_summary,
            retrieved_docs=[],  # tools handle retrieval
            target_tokens=target_tokens,
        )

        # Build LLM with tools bound
        all_tools = get_tools()
        # Filter to domain tools + any MCP tools
        domain_tool_names = set(domain_tools)
        active_tools = [t for t in all_tools if t.name in domain_tool_names] + [
            t
            for t in all_tools
            if t.name
            not in {
                "knowledge_base_search",
                "billing_search",
                "technical_search",
                "compliance_search",
            }
        ]

        llm = _get_llm(model_key)
        llm_with_tools = llm.bind_tools(active_tools)

        # Build message list
        full_messages = [SystemMessage(content=system_content)] + list(messages)
        if not any(isinstance(m, HumanMessage) and m.content == task for m in messages[-3:]):
            full_messages.append(HumanMessage(content=task))

        response: AIMessage = llm_with_tools.invoke(full_messages)

        logger.info(
            f"{agent_name}_agent_response",
            has_tool_calls=bool(response.tool_calls),
            tool_names=[tc["name"] for tc in (response.tool_calls or [])],
        )

        return {
            **state,
            "messages": [response],
            "iteration": state.get("iteration", 0) + 1,
        }

    agent_node.__name__ = f"{agent_name}_agent_node"
    return agent_node


# ── Domain subgraph agent nodes ───────────────────────────────────────────────
billing_agent_node = _build_agent_node(
    agent_name="billing",
    model_key="billing_model",
    system_prompt_name="billing_agent",
    domain_tools=["billing_search", "knowledge_base_search"],
)

technical_agent_node = _build_agent_node(
    agent_name="technical",
    model_key="technical_model",
    system_prompt_name="technical_agent",
    domain_tools=["technical_search", "knowledge_base_search"],
)

compliance_agent_node = _build_agent_node(
    agent_name="compliance",
    model_key="compliance_model",
    system_prompt_name="compliance_agent",
    domain_tools=["compliance_search", "knowledge_base_search"],
)

general_agent_node = _build_agent_node(
    agent_name="general",
    model_key="default_model",
    system_prompt_name="general_agent",
    domain_tools=[
        "knowledge_base_search",
        "billing_search",
        "technical_search",
        "compliance_search",
    ],
)


# ── Escalation node ───────────────────────────────────────────────────────────
@traceable(name="node.escalation")
def escalation_node(state: AgentState) -> AgentState:
    """Handle escalation — acknowledge urgency and provide escalation path."""
    task = state.get("task", "")
    logger.warning("escalation_triggered", task=task[:100])
    return {
        **state,
        "final_answer": (
            "⚠️ **This request has been escalated for urgent human review.**\n\n"
            "I've detected that your situation requires immediate attention from our team. "
            "A support specialist will contact you within 2 business hours.\n\n"
            "**Reference ID:** will be provided by your support representative.\n\n"
            "If this is a genuine emergency, please call our priority support line directly."
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
#  Tool node
# ────────────────────────────────────────────────────────────────────────────


@traceable(name="node.tool")
def tool_node(state: AgentState) -> AgentState:
    """Execute tool calls from the last AI message and append results."""
    from src.tools.registry import get_tools_by_name

    messages = state.get("messages", [])
    if not messages:
        return state

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return state

    tools_by_name = get_tools_by_name()
    tool_results: list[ToolMessage] = []
    sources: list[str] = list(state.get("sources", []))

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        tool = tools_by_name.get(tool_name)
        if tool is None:
            result = f"Error: Tool '{tool_name}' not found."
            logger.warning("tool_not_found", tool_name=tool_name)
        else:
            try:
                result = tool.invoke(tool_args)
                logger.info("tool_executed", tool_name=tool_name, args=str(tool_args)[:80])

                # Track search queries for citation trail
                if "search" in tool_name and isinstance(tool_args, dict):
                    question = tool_args.get("question", "")
                    if question:
                        sources.append(f"{tool_name}:{question}")

            except Exception as exc:
                result = f"Error executing {tool_name}: {exc!s}"
                logger.error("tool_execution_error", tool_name=tool_name, error=str(exc))

        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
            )
        )

    return {
        **state,
        "messages": tool_results,
        "sources": sources,
    }


# ────────────────────────────────────────────────────────────────────────────
#  Final answer extraction node
# ────────────────────────────────────────────────────────────────────────────


@traceable(name="node.extract_final_answer", run_type="llm")
def extract_final_answer_node(state: AgentState) -> AgentState:
    """Extract the final text answer from the last AI message."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return {**state, "final_answer": str(msg.content)}
    return state
