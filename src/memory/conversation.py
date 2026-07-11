"""
src/memory/conversation.py — Rolling conversation summariser.

Keeps a running summary of the conversation in LangGraph state rather than
storing every message, which keeps context windows small while preserving
long-term coherence.

Design
------
* ``ConversationSummaryMemory`` is intentionally in-memory / stateless.
  The caller (graph node) is responsible for persisting the returned summary
  string into graph state between turns.
* The Gemini client is created once per process via ``@lru_cache`` on the
  factory helper to avoid repeated SDK initialisation overhead.
* If ``prompts/summarizer_system.md`` is missing the class falls back to an
  embedded hardcoded prompt so the service can still boot without the file.

Usage::

    from src.memory.conversation import ConversationSummaryMemory
    from langchain_core.messages import HumanMessage, AIMessage

    mem = ConversationSummaryMemory(thread_id="thread-abc123")

    new_messages = [
        HumanMessage(content="My name is Alice and I have chest pain."),
        AIMessage(content="I'm sorry to hear that. How long have you had the pain?"),
    ]

    summary = await mem.summarize_history(
        previous_summary=None,
        new_messages=new_messages,
    )
    # summary -> "Patient Alice presented with chest pain of unspecified duration."
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_SUMMARIZER_PROMPT_PATH = _BASE_DIR / "prompts" / "summarizer_system.md"

# ── Fallback prompt ───────────────────────────────────────────────────────────
_FALLBACK_SYSTEM_PROMPT = (
    "You are a clinical conversation summariser for an intelligent medical triage "
    "and billing assistant.\n\n"
    "Your job is to produce a concise, factual, third-person summary of the "
    "conversation so far.  The summary must:\n"
    "  • Preserve all medically and administratively relevant facts "
    "(symptoms, diagnoses, dates, insurance details, patient identifiers, etc.).\n"
    "  • Incorporate the *previous summary* (if provided) so that context from "
    "earlier turns is not lost.\n"
    "  • Be written in plain prose — no bullet points, no headers.\n"
    "  • Omit pleasantries, filler words, and repeated information.\n\n"
    "Return ONLY the updated summary text.  Do not add any preamble, "
    'explanation, or labels such as "Summary:".'
)


def _load_system_prompt() -> str:
    """Load the summariser system prompt from disk, with a hardcoded fallback."""
    if _SUMMARIZER_PROMPT_PATH.exists():
        try:
            content = _SUMMARIZER_PROMPT_PATH.read_text(encoding="utf-8").strip()
            if content:
                logger.debug(
                    "conversation.prompt_loaded",
                    path=str(_SUMMARIZER_PROMPT_PATH),
                )
                return content
        except OSError as exc:
            logger.warning(
                "conversation.prompt_load_error",
                path=str(_SUMMARIZER_PROMPT_PATH),
                error=str(exc),
            )

    logger.info(
        "conversation.prompt_fallback",
        reason="file_not_found_or_empty",
        path=str(_SUMMARIZER_PROMPT_PATH),
    )
    return _FALLBACK_SYSTEM_PROMPT


@lru_cache(maxsize=1)
def _get_llm():
    """Create and cache the Gemini LLM client (once per process).

    Returns
    -------
    ChatGoogleGenerativeAI
        A LangChain-wrapped Gemini 2.5 Flash model configured for
        summarisation tasks (low temperature, deterministic outputs).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

    from src.config import settings  # noqa: PLC0415

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key or None,
        temperature=0.0,
        max_retries=3,
    )
    logger.info("conversation.llm_created", model="gemini-2.5-flash")
    return llm


def _format_messages(messages: list[Any]) -> str:
    """Format a list of LangChain message objects as ``Role: Content`` lines.

    Accepts both LangChain ``BaseMessage`` instances and plain dicts with
    ``role`` / ``content`` keys so that callers are not forced to import
    LangChain message types just to call this function.

    Parameters
    ----------
    messages:
        A list of ``HumanMessage``, ``AIMessage``, ``SystemMessage`` objects
        *or* plain dicts with ``role`` and ``content`` keys.

    Returns
    -------
    str
        Multi-line string ready to be embedded in the summarisation prompt.
    """
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown").capitalize()
            content = str(msg.get("content", ""))
        else:
            # LangChain BaseMessage subclasses expose .type and .content
            type_to_role: dict[str, str] = {
                "human": "User",
                "ai": "Assistant",
                "system": "System",
                "tool": "Tool",
                "function": "Function",
            }
            msg_type = getattr(msg, "type", "unknown")
            role = type_to_role.get(msg_type, msg_type.capitalize())
            content = str(getattr(msg, "content", ""))

        # Collapse newlines within a single message to keep the block readable.
        content = content.replace("\n", " ").strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


class ConversationSummaryMemory:
    """Rolling conversation summariser for LangGraph-based RAG pipelines.

    This class is intentionally **stateless** — the caller stores the
    returned summary string in graph state and passes it back on the next
    invocation.  This makes the object cheap to instantiate per-turn.

    Parameters
    ----------
    thread_id:
        The active conversation thread identifier.  Used only for logging.
    """

    def __init__(self, thread_id: str) -> None:
        self.thread_id: str = thread_id
        self._system_prompt: str = _load_system_prompt()

        logger.debug("conversation.init", thread_id=thread_id)

    # ── Public API ────────────────────────────────────────────────────────────

    async def summarize_history(
        self,
        previous_summary: str | None,
        new_messages: list[Any],
    ) -> str:
        """Produce an updated rolling summary of the conversation.

        The LLM is given the previous summary (if any) plus the most recent
        batch of messages and is asked to merge them into a single updated
        summary.

        Parameters
        ----------
        previous_summary:
            The summary from the previous turn, or ``None`` / empty string
            if this is the first summarisation call.
        new_messages:
            A list of LangChain ``BaseMessage`` objects (or compatible dicts)
            representing the conversation turns since the last summary.

        Returns
        -------
        str
            The updated summary string.  Empty string if ``new_messages`` is
            empty and no previous summary exists.
        """
        if not new_messages and not previous_summary:
            logger.debug(
                "conversation.summarize.noop",
                thread_id=self.thread_id,
                reason="no_messages_no_previous_summary",
            )
            return ""

        formatted_messages = _format_messages(new_messages) if new_messages else ""

        # Build the user content block
        user_parts: list[str] = []
        if previous_summary and previous_summary.strip():
            user_parts.append(f"PREVIOUS SUMMARY:\n{previous_summary.strip()}")
        if formatted_messages:
            user_parts.append(f"NEW MESSAGES:\n{formatted_messages}")
        user_parts.append(
            "Please produce an updated summary that incorporates all the above information."
        )
        user_content = "\n\n".join(user_parts)

        logger.info(
            "conversation.summarize.start",
            thread_id=self.thread_id,
            new_message_count=len(new_messages),
            has_previous_summary=bool(previous_summary),
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

            llm = _get_llm()
            lc_messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=user_content),
            ]

            # Support both sync and async LLM invocations transparently.
            if asyncio.iscoroutinefunction(llm.ainvoke):
                response = await llm.ainvoke(lc_messages)
            else:
                # Wrap synchronous call in a thread pool to stay non-blocking.
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, llm.invoke, lc_messages)

            summary: str = str(getattr(response, "content", response)).strip()

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "conversation.summarize.error",
                thread_id=self.thread_id,
                error=str(exc),
                exc_info=True,
            )
            # Graceful degradation: return the previous summary rather than
            # crashing the graph node.
            return previous_summary or ""

        logger.info(
            "conversation.summarize.done",
            thread_id=self.thread_id,
            summary_length=len(summary),
        )
        return summary

    def summarize_history_sync(
        self,
        previous_summary: str | None,
        new_messages: list[Any],
    ) -> str:
        """Synchronous wrapper around :meth:`summarize_history`.

        Useful in contexts where an event loop is not running (e.g. unit
        tests or CLI scripts).

        Parameters
        ----------
        previous_summary:
            See :meth:`summarize_history`.
        new_messages:
            See :meth:`summarize_history`.

        Returns
        -------
        str
            The updated summary string.
        """
        return asyncio.run(self.summarize_history(previous_summary, new_messages))

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ConversationSummaryMemory(thread_id={self.thread_id!r})"
