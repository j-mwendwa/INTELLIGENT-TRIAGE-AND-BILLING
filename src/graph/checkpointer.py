"""
src/graph/checkpointer.py — High-level turn wrappers.

run_turn()        — sync (scripts, tests)
run_turn_stream() — async generator yielding SSE events (FastAPI streaming)
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import structlog

from src.graph.state import AgentState

logger = structlog.get_logger(__name__)


def _make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _build_initial_state(
    task: str,
    thread_id: str,
    entity_memory: dict | None = None,
) -> AgentState:
    from src.config import cfg
    from src.memory.entity_memory import EntityMemory

    mem = EntityMemory(thread_id)
    merged_memory = {**mem.all(), **(entity_memory or {})}

    return AgentState(
        messages=[],
        task=task,
        entity_memory=merged_memory,
        conversation_summary=None,
        sources=[],
        intent=None,
        routing_confidence=None,
        active_subagent=None,
        input_security=None,
        final_answer=None,
        iteration=0,
        max_iterations=cfg._data.get("graph", {}).get("max_iterations", 8),
    )


# ── Sync (tests / scripts) ────────────────────────────────────────────────────


def run_turn(
    task: str,
    thread_id: str | None = None,
    entity_memory: dict | None = None,
) -> dict:
    """Run a single conversation turn synchronously. Returns final state."""
    from src.graph.graph import get_app

    thread_id = thread_id or str(uuid.uuid4())
    app = get_app()
    state = _build_initial_state(task, thread_id, entity_memory)

    logger.info("run_turn_start", thread_id=thread_id, task=task[:60])
    final_state = app.invoke(state, config=_make_config(thread_id))
    logger.info("run_turn_done", thread_id=thread_id, intent=final_state.get("intent"))
    return final_state


# ── Async streaming (FastAPI) ─────────────────────────────────────────────────


async def run_turn_stream(
    task: str,
    thread_id: str | None = None,
    entity_memory: dict | None = None,
) -> AsyncIterator[dict]:
    """
    Async generator yielding SSE event dicts:
      {"type": "token", "data": "..."}
      {"type": "node", "data": {"node": "...", "intent": "..."}}
      {"type": "done", "data": {"final_answer": "...", "intent": "...", "sources": [...]}}
    """
    from src.graph.graph import get_app_async

    thread_id = thread_id or str(uuid.uuid4())
    app = await get_app_async()
    state = _build_initial_state(task, thread_id, entity_memory)

    logger.info("run_turn_stream_start", thread_id=thread_id, task=task[:60])

    final_state: dict = {}

    async for event in app.astream(state, config=_make_config(thread_id), stream_mode="updates"):
        for node_name, node_output in event.items():
            if not isinstance(node_output, dict):
                continue

            # Yield node progress event
            yield {
                "type": "node",
                "data": {
                    "node": node_name,
                    "intent": node_output.get("intent"),
                },
            }

            # Stream tokens from final_answer if available
            if answer := node_output.get("final_answer"):
                # Stream in 20-char chunks to simulate token streaming
                chunk_size = 20
                for i in range(0, len(answer), chunk_size):
                    yield {"type": "token", "data": answer[i : i + chunk_size]}
                    await asyncio.sleep(0)  # yield control

            final_state.update(node_output)

    yield {
        "type": "done",
        "data": {
            "final_answer": final_state.get("final_answer", ""),
            "intent": final_state.get("intent"),
            "active_subagent": final_state.get("active_subagent"),
            "sources": final_state.get("sources", []),
            "routing_confidence": final_state.get("routing_confidence"),
            "thread_id": thread_id,
        },
    }

    logger.info(
        "run_turn_stream_done",
        thread_id=thread_id,
        intent=final_state.get("intent"),
    )
