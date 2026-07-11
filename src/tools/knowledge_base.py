"""
src/tools/knowledge_base.py — Domain-specific RAG search tools.

Each @tool maps to a separate collection in the vector store.
The supervisor agent can call any of these; subgraph agents call their own.
"""

from __future__ import annotations

import structlog
from langchain_core.tools import tool

from src.config import cfg

logger = structlog.get_logger(__name__)


def _search_collection(question: str, collection: str) -> str:
    """Shared retrieval logic for any collection."""
    from src.retrieval.llamaindex_retriever import LlamaIndexRetriever

    retriever = LlamaIndexRetriever(collection=collection)
    results = retriever.retrieve(question)
    if not results:
        return f"No relevant documents found in {collection} for: {question}"

    formatted = []
    for i, node in enumerate(results, 1):
        score = getattr(node, "score", None)
        score_str = f" (score: {score:.3f})" if score is not None else ""
        formatted.append(f"[{i}]{score_str}\n{node.text}")

    return "\n\n---\n\n".join(formatted)


@tool
def knowledge_base_search(question: str) -> str:
    """Search the general knowledge base for information relevant to the question.

    Use this tool for general queries that don't clearly belong to billing,
    technical, or compliance domains.
    """
    logger.info("kb_search", question=question[:100], collection="knowledge_base")
    return _search_collection(question, "knowledge_base")


@tool
def billing_search(question: str) -> str:
    """Search the billing knowledge base for invoice, payment, refund, or pricing information.

    Use this for any query related to:
    - Invoice disputes or billing errors
    - Payment methods and processing
    - Refund requests and policies
    - Pricing, quotes, and rate schedules
    - Account credits and adjustments
    - Late fees and penalties
    """
    collection = cfg.retrieval.get("billing_collection", "billing_kb")
    logger.info("billing_search", question=question[:100], collection=collection)
    return _search_collection(question, collection)


@tool
def technical_search(question: str) -> str:
    """Search the technical knowledge base for service issues, troubleshooting, or system errors.

    Use this for any query related to:
    - Service outages or degraded performance
    - Error messages and troubleshooting steps
    - API integration issues
    - Feature requests and product limitations
    - Configuration and setup guidance
    - SLA and uptime commitments
    """
    collection = cfg.retrieval.get("technical_collection", "technical_kb")
    logger.info("technical_search", question=question[:100], collection=collection)
    return _search_collection(question, collection)


@tool
def compliance_search(question: str) -> str:
    """Search the compliance knowledge base for regulatory, policy, or legal information.

    Use this for any query related to:
    - Data privacy and GDPR/CCPA compliance
    - Terms of service and acceptable use policies
    - Security certifications (SOC2, ISO27001)
    - Regulatory requirements and audits
    - Data retention and deletion policies
    - Escalation procedures for legal matters
    """
    collection = cfg.retrieval.get("compliance_collection", "compliance_kb")
    logger.info("compliance_search", question=question[:100], collection=collection)
    return _search_collection(question, collection)
