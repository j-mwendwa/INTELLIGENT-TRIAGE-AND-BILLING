"""
evals/run_evals.py — RAGAS / DeepEval evaluation pipelines.

Usage:
    python evals/run_evals.py --pipeline rag
    python evals/run_evals.py --pipeline agent
    python evals/run_evals.py --pipeline agent --output evals/results/latest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Evaluation dataset ────────────────────────────────────────────────────────
EVAL_DATASET = [
    {
        "question": "What is the refund policy for cancelled subscriptions?",
        "domain": "billing",
        "expected_keywords": ["refund", "cancel", "days", "policy"],
    },
    {
        "question": "How do I troubleshoot API 503 errors?",
        "domain": "technical",
        "expected_keywords": ["503", "error", "troubleshoot", "retry"],
    },
    {
        "question": "What are your GDPR data retention obligations?",
        "domain": "compliance",
        "expected_keywords": ["GDPR", "data", "retention", "rights"],
    },
    {
        "question": "I was double-charged on invoice INV-2024-100. Help?",
        "domain": "billing",
        "expected_keywords": ["invoice", "charge", "dispute", "refund"],
    },
    {
        "question": "Do you have SOC2 Type II certification?",
        "domain": "compliance",
        "expected_keywords": ["SOC2", "certification", "audit"],
    },
    {
        "question": "Our webhook endpoint stopped receiving events yesterday.",
        "domain": "technical",
        "expected_keywords": ["webhook", "event", "endpoint"],
    },
]


def run_rag_pipeline(verbose: bool = False) -> dict:
    """Evaluate raw retrieval quality (without the full agent graph)."""
    from src.core.llamaindex_setup import setup_llamaindex
    from src.retrieval.llamaindex_retriever import LlamaIndexRetriever

    setup_llamaindex()

    results = []
    for item in EVAL_DATASET:
        collection_map = {
            "billing": "billing_kb",
            "technical": "technical_kb",
            "compliance": "compliance_kb",
        }
        collection = collection_map.get(item["domain"], "knowledge_base")

        try:
            retriever = LlamaIndexRetriever(collection=collection)
            nodes = retriever.retrieve(item["question"])
            retrieved_text = " ".join(n.text for n in nodes).lower()

            hits = [kw for kw in item["expected_keywords"] if kw.lower() in retrieved_text]
            recall = (
                len(hits) / len(item["expected_keywords"]) if item["expected_keywords"] else 0.0
            )

            result = {
                "question": item["question"],
                "domain": item["domain"],
                "nodes_retrieved": len(nodes),
                "keyword_recall": recall,
                "keywords_hit": hits,
                "status": "ok",
            }
        except Exception as exc:
            result = {
                "question": item["question"],
                "domain": item["domain"],
                "status": "error",
                "error": str(exc),
                "keyword_recall": 0.0,
            }

        results.append(result)
        if verbose:
            print(
                f"  [{item['domain']}] {item['question'][:50]}… → recall={result['keyword_recall']:.0%}"
            )

    avg_recall = sum(r["keyword_recall"] for r in results) / len(results)
    return {"pipeline": "rag", "results": results, "avg_keyword_recall": avg_recall}


def run_agent_pipeline(verbose: bool = False) -> dict:
    """Evaluate the full agent graph (supervisor + subgraphs)."""
    from src.core.llamaindex_setup import setup_llamaindex
    from src.graph.checkpointer import run_turn
    from src.tools.registry import register_base_tools

    setup_llamaindex()
    register_base_tools()

    results = []
    for item in EVAL_DATASET:
        start = time.perf_counter()
        try:
            state = run_turn(task=item["question"])
            elapsed = time.perf_counter() - start

            answer = (state.get("final_answer") or "").lower()
            intent = state.get("intent", "unknown")
            hits = [kw for kw in item["expected_keywords"] if kw.lower() in answer]
            recall = (
                len(hits) / len(item["expected_keywords"]) if item["expected_keywords"] else 0.0
            )
            routing_correct = intent == item["domain"]

            result = {
                "question": item["question"],
                "domain": item["domain"],
                "detected_intent": intent,
                "routing_correct": routing_correct,
                "keyword_recall": recall,
                "keywords_hit": hits,
                "latency_s": round(elapsed, 2),
                "status": "ok",
            }
        except Exception as exc:
            elapsed = time.perf_counter() - start
            result = {
                "question": item["question"],
                "domain": item["domain"],
                "status": "error",
                "error": str(exc),
                "keyword_recall": 0.0,
                "routing_correct": False,
                "latency_s": round(elapsed, 2),
            }

        results.append(result)
        if verbose:
            status = "✅" if result.get("routing_correct") else "❌"
            print(
                f"  {status} [{item['domain']}→{result.get('detected_intent', '?')}] "
                f"{item['question'][:45]}… recall={result['keyword_recall']:.0%} "
                f"({result['latency_s']}s)"
            )

    avg_recall = sum(r["keyword_recall"] for r in results) / len(results)
    routing_acc = sum(1 for r in results if r.get("routing_correct")) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)

    return {
        "pipeline": "agent",
        "results": results,
        "avg_keyword_recall": avg_recall,
        "routing_accuracy": routing_acc,
        "avg_latency_s": round(avg_latency, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG or agent evaluations.")
    parser.add_argument("--pipeline", choices=["rag", "agent"], required=True)
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from src.config import settings
    from src.core.logging import setup_logging

    setup_logging("WARNING", settings.app_env)

    print(f"\n🔬 Running {args.pipeline.upper()} evaluation pipeline...")
    print(f"   Questions: {len(EVAL_DATASET)}\n")

    if args.pipeline == "rag":
        report = run_rag_pipeline(verbose=args.verbose)
        print(f"\n📊 Average keyword recall: {report['avg_keyword_recall']:.1%}")
    else:
        report = run_agent_pipeline(verbose=args.verbose)
        print(f"\n📊 Routing accuracy   : {report['routing_accuracy']:.1%}")
        print(f"   Avg keyword recall : {report['avg_keyword_recall']:.1%}")
        print(f"   Avg latency        : {report['avg_latency_s']}s")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
