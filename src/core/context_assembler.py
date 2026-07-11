"""
src/core/context_assembler.py
─────────────────────────────
Token-aware context assembly for the Intelligent Triage & Billing RAG system.

``ContextAssembler`` takes the individual parts of a prompt (system prompt,
entity memory, conversation summary, retrieved documents) and combines them
into a single context string that fits within a target token budget.

Trimming strategy
-----------------
Retrieved documents are considered **the most expendable** content because:
* They can always be re-retrieved.
* System prompts and entity memory contain policy / patient-specific data that
  is harder to reconstruct on the fly.

When the assembled context exceeds ``target_tokens``, documents are removed
from the *back* of the list (least-relevant first, assuming the retriever
returns results in descending relevance order) until the budget is satisfied.

Usage
-----
    from src.core.token_counter import TokenCounter
    from src.core.context_assembler import ContextAssembler

    assembler = ContextAssembler(token_counter=TokenCounter())
    context = assembler.build(
        system_prompt=system_prompt,
        entity_memory={"patient_id": "P-001", "insurance": "Aetna"},
        conversation_summary="Patient called about claim #12345.",
        retrieved_docs=["Doc A text ...", "Doc B text ..."],
        target_tokens=8000,
    )
"""

from __future__ import annotations

from typing import Any

import structlog

from src.core.token_counter import TokenCounter

logger = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# XML-tag delimiters used in the assembled context string
# ──────────────────────────────────────────────────────────────────────────────

_TAG_ENTITY_MEMORY_OPEN = "<entity_memory>"
_TAG_ENTITY_MEMORY_CLOSE = "</entity_memory>"
_TAG_CONV_SUMMARY_OPEN = "<conversation_summary>"
_TAG_CONV_SUMMARY_CLOSE = "</conversation_summary>"
_TAG_RETRIEVED_DOCS_OPEN = "<retrieved_documents>"
_TAG_RETRIEVED_DOCS_CLOSE = "</retrieved_documents>"
_TAG_DOC_OPEN = '<document index="{idx}">'
_TAG_DOC_CLOSE = "</document>"

# Number of tokens reserved for the XML tags and separators in the
# entity-memory and conversation-summary blocks (rough estimate).
_TAG_OVERHEAD_TOKENS: int = 20


class ContextAssembler:
    """
    Assemble a token-bounded context string from prompt components.

    Parameters
    ----------
    token_counter:
        A ``TokenCounter`` instance used to measure each section.
        Inject a shared instance to benefit from the cached encoding.
    """

    def __init__(self, token_counter: TokenCounter) -> None:
        self._tc: TokenCounter = token_counter

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(
        self,
        system_prompt: str,
        entity_memory: dict[str, Any],
        conversation_summary: str | None,
        retrieved_docs: list[str],
        target_tokens: int = 8000,
    ) -> str:
        """
        Build a context string from the supplied components.

        Assembly order
        --------------
        1. System prompt (never trimmed — trim retrieved docs first)
        2. ``<entity_memory>`` block  (key: value pairs)
        3. ``<conversation_summary>`` block  (omitted if ``None`` / empty)
        4. ``<retrieved_documents>`` block   (trimmed from the back as needed)

        Parameters
        ----------
        system_prompt:
            The top-level instruction prompt for the LLM.
        entity_memory:
            Dictionary of extracted entities (e.g. patient ID, insurer,
            claim number).  Rendered as ``key: value`` lines.
        conversation_summary:
            A running summary of the conversation so far.  Pass ``None`` or
            an empty string to omit this block entirely.
        retrieved_docs:
            List of retrieved document passages, ordered by relevance
            (most relevant first).  Passages are trimmed from the *back* when
            the budget is exceeded.
        target_tokens:
            Maximum token count for the assembled context string.
            Defaults to 8 000.

        Returns
        -------
        str
            The assembled context string, guaranteed to be ≤ ``target_tokens``
            tokens (modulo the ``~5 %`` tiktoken / Gemini approximation).
        """
        # ── 1. Build static sections ───────────────────────────────────────────
        system_block = system_prompt.strip()
        entity_block = self._render_entity_memory(entity_memory)
        summary_block = self._render_conversation_summary(conversation_summary)

        # ── 2. Count static token usage ────────────────────────────────────────
        static_tokens = (
            self._tc.count(system_block)
            + self._tc.count(entity_block)
            + self._tc.count(summary_block)
        )

        # ── 3. Fit retrieved docs into the remaining budget ────────────────────
        docs_budget = max(0, target_tokens - static_tokens)
        trimmed_docs, docs_tokens, docs_dropped = self._fit_docs(retrieved_docs, docs_budget)
        docs_block = self._render_retrieved_docs(trimmed_docs)

        # ── 4. Assemble final context ──────────────────────────────────────────
        parts: list[str] = [system_block]
        if entity_block:
            parts.append(entity_block)
        if summary_block:
            parts.append(summary_block)
        if docs_block:
            parts.append(docs_block)

        context = "\n\n".join(parts)
        total_tokens = self._tc.count(context)

        # ── 5. Log breakdown ───────────────────────────────────────────────────
        context_breakdown = {
            "system_prompt_tokens": self._tc.count(system_block),
            "entity_memory_tokens": self._tc.count(entity_block),
            "conversation_summary_tokens": self._tc.count(summary_block),
            "retrieved_docs_tokens": docs_tokens,
            "docs_included": len(trimmed_docs),
            "docs_dropped": docs_dropped,
            "total_context_tokens": total_tokens,
            "target_tokens": target_tokens,
            "over_budget": total_tokens > target_tokens,
        }
        logger.info("context_breakdown", **context_breakdown)

        if total_tokens > target_tokens:
            logger.warning(
                "context_over_budget",
                total=total_tokens,
                target=target_tokens,
                excess=total_tokens - target_tokens,
            )

        return context

    # ── Rendering helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _render_entity_memory(entity_memory: dict[str, Any]) -> str:
        """Render the entity-memory dict as an XML-tagged block."""
        if not entity_memory:
            return ""
        lines = "\n".join(f"  {key}: {value}" for key, value in entity_memory.items())
        return f"{_TAG_ENTITY_MEMORY_OPEN}\n{lines}\n{_TAG_ENTITY_MEMORY_CLOSE}"

    @staticmethod
    def _render_conversation_summary(summary: str | None) -> str:
        """Render the conversation summary as an XML-tagged block, or '' if absent."""
        if not summary or not summary.strip():
            return ""
        return f"{_TAG_CONV_SUMMARY_OPEN}\n{summary.strip()}\n{_TAG_CONV_SUMMARY_CLOSE}"

    def _render_retrieved_docs(self, docs: list[str]) -> str:
        """Wrap each retrieved passage in a numbered ``<document>`` tag."""
        if not docs:
            return ""
        doc_parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            open_tag = _TAG_DOC_OPEN.format(idx=idx)
            doc_parts.append(f"{open_tag}\n{doc.strip()}\n{_TAG_DOC_CLOSE}")
        inner = "\n\n".join(doc_parts)
        return f"{_TAG_RETRIEVED_DOCS_OPEN}\n{inner}\n{_TAG_RETRIEVED_DOCS_CLOSE}"

    # ── Budget management ──────────────────────────────────────────────────────

    def _fit_docs(
        self,
        docs: list[str],
        budget_tokens: int,
    ) -> tuple[list[str], int, int]:
        """
        Greedily include documents (most relevant first) within *budget_tokens*.

        Parameters
        ----------
        docs:
            Document passages in relevance order (most relevant at index 0).
        budget_tokens:
            Maximum token allowance for the documents block.

        Returns
        -------
        tuple[list[str], int, int]
            * The trimmed list of included documents.
            * Total token count of the included documents.
            * Number of documents dropped.
        """
        if not docs or budget_tokens <= 0:
            return [], 0, len(docs)

        included: list[str] = []
        used_tokens: int = 0
        # Account for the outer wrapper tags.
        wrapper_tokens = self._tc.count(f"{_TAG_RETRIEVED_DOCS_OPEN}\n{_TAG_RETRIEVED_DOCS_CLOSE}")
        remaining = budget_tokens - wrapper_tokens

        for doc in docs:
            # Estimate tokens for this document including its XML tags.
            doc_rendered = (
                f"{_TAG_DOC_OPEN.format(idx=len(included) + 1)}\n{doc.strip()}\n{_TAG_DOC_CLOSE}"
            )
            doc_tokens = self._tc.count(doc_rendered)
            if used_tokens + doc_tokens > remaining:
                # This doc would push us over budget — skip it and all
                # subsequent docs (they're less relevant anyway).
                break
            included.append(doc)
            used_tokens += doc_tokens

        dropped = len(docs) - len(included)
        if dropped:
            logger.info(
                "retrieved_docs_trimmed",
                total=len(docs),
                included=len(included),
                dropped=dropped,
                budget_tokens=budget_tokens,
                used_tokens=used_tokens,
            )
        return included, used_tokens, dropped
