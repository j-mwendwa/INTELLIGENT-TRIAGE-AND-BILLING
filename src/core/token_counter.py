"""
src/core/token_counter.py
─────────────────────────
tiktoken-based token counting utilities for the Intelligent Triage & Billing
RAG system.

Why tiktoken for Gemini?
------------------------
Google's Gemini models do not expose a public tokenizer library.  The
``cl100k_base`` BPE vocabulary (used by GPT-4 / GPT-3.5) produces counts that
are within **~5 %** of Gemini's actual token budget, which is accurate enough
for context-window management.  Always add a small safety margin (e.g. 5 %)
on top of any budget limit when using these counts for Gemini.

Usage
-----
    from src.core.token_counter import TokenCounter

    counter = TokenCounter()                        # default: cl100k_base
    n = counter.count("Hello, world!")              # → 4
    n = counter.count_messages(messages)            # LangChain BaseMessage list
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog
import tiktoken

logger = structlog.get_logger(__name__)

# Sentinel value returned when encoding fails so callers receive a safe
# over-estimate rather than crashing.
_FALLBACK_CHARS_PER_TOKEN: int = 4


@lru_cache(maxsize=8)
def _get_encoding(model_or_encoding: str) -> tiktoken.Encoding:
    """
    Return a cached tiktoken ``Encoding`` for *model_or_encoding*.

    Tries ``tiktoken.encoding_for_model`` first (for model names like
    ``"gpt-4"``), then falls back to ``tiktoken.get_encoding`` (for encoding
    names like ``"cl100k_base"``).
    """
    try:
        return tiktoken.encoding_for_model(model_or_encoding)
    except KeyError:
        return tiktoken.get_encoding(model_or_encoding)


class TokenCounter:
    """
    Lightweight wrapper around tiktoken for estimating token usage.

    Parameters
    ----------
    model:
        A tiktoken encoding name (e.g. ``"cl100k_base"``) or an OpenAI model
        name (e.g. ``"gpt-4"``).  Defaults to ``"cl100k_base"``, which
        approximates Gemini token counts within ~5 %.

    Notes
    -----
    * Instances share an LRU-cached encoding object, so creating multiple
      ``TokenCounter`` instances for the same model is effectively free.
    * All public methods return ``int`` and never raise; encoding errors are
      caught and a character-length estimate is returned instead.

    Gemini accuracy note
    --------------------
    Google Gemini uses a SentencePiece-based tokenizer that is not publicly
    distributed.  ``cl100k_base`` BPE counts are within **~5 %** of Gemini's
    actual token budget.  When using these counts to enforce context-window
    limits for Gemini, apply a 5 % safety margin::

        budget_tokens = int(model_context_limit * 0.95)
    """

    def __init__(self, model: str = "cl100k_base") -> None:
        self.model: str = model
        self._encoding: tiktoken.Encoding = _get_encoding(model)
        logger.debug("token_counter_initialised", model=model)

    # ── Primary API ───────────────────────────────────────────────────────────

    def count(self, text: str) -> int:
        """
        Count the number of tokens in *text*.

        Parameters
        ----------
        text:
            Plain-text string to tokenize.

        Returns
        -------
        int
            Token count.  Returns a character-length / 4 estimate if the
            tiktoken encoding raises an unexpected error.
        """
        if not text:
            return 0
        try:
            return len(self._encoding.encode(text))
        except Exception as exc:  # pragma: no cover
            estimate = max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)
            logger.warning(
                "token_count_fallback",
                error=str(exc),
                text_len=len(text),
                estimate=estimate,
            )
            return estimate

    def count_messages(self, messages: list[Any]) -> int:
        """
        Count tokens across a list of LangChain ``BaseMessage`` objects.

        Each message contributes:
        * The token count of its ``content`` (string or stringified).
        * A fixed overhead of **4** tokens per message to account for the
          role separator tokens that chat-format models inject (``<|im_start|>``,
          role name, ``<|im_sep|>``, etc.).
        * A final **2**-token overhead for the reply primer (``<|im_start|>``
          + ``assistant``).

        Parameters
        ----------
        messages:
            A list of objects with a ``.content`` attribute (LangChain
            ``BaseMessage`` subclasses), or plain strings.  Mixed lists are
            handled gracefully.

        Returns
        -------
        int
            Estimated total token count for the entire message sequence.

        Notes
        -----
        For Gemini models, add ~5 % to the returned value as a safety margin
        because the tokenizer differs from ``cl100k_base``.
        """
        if not messages:
            return 0

        _PER_MESSAGE_OVERHEAD = 4  # role / separator tokens
        _REPLY_PRIMER = 2  # <|im_start|>assistant primer

        total = _REPLY_PRIMER
        for msg in messages:
            content = self._extract_content(msg)
            total += self.count(content) + _PER_MESSAGE_OVERHEAD

        logger.debug(
            "count_messages_complete",
            num_messages=len(messages),
            total_tokens=total,
        )
        return total

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_content(message: Any) -> str:
        """
        Pull the text content out of *message* regardless of its type.

        Handles:
        * LangChain ``BaseMessage`` (has ``.content`` attribute).
        * Plain ``str``.
        * Anything else: falls back to ``str(message)``.
        """
        if isinstance(message, str):
            return message
        content = getattr(message, "content", None)
        if content is None:
            return str(message)
        if isinstance(content, str):
            return content
        # Multimodal content is a list of dicts; extract text parts only.
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return str(content)
