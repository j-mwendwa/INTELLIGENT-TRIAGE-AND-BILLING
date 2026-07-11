#!/usr/bin/env python3
"""
scripts/audit_context.py — Print the full LLM context for a given message.

Usage:
    python scripts/audit_context.py --message "Why was I charged $150?"
    python scripts/audit_context.py --message "API 503 error" --thread-id abc123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit what the LLM sees for a given message.",
    )
    parser.add_argument("--message", required=True, help="The user message to audit")
    parser.add_argument("--thread-id", default=None, help="Thread ID to use for memory context")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from src.config import settings
    from src.core.logging import setup_logging

    setup_logging(log_level="WARNING", app_env=settings.app_env)

    print("\n" + "═" * 60)
    print("  INTELLIGENT TRIAGE & BILLING — CONTEXT AUDIT")
    print("═" * 60)
    print(f"  Message   : {args.message}")
    print(f"  Thread ID : {args.thread_id or '(new thread)'}")
    print("═" * 60 + "\n")

    # Setup LlamaIndex
    from src.core.llamaindex_setup import setup_llamaindex

    setup_llamaindex()

    # Register tools
    from src.tools.registry import register_base_tools

    register_base_tools()

    # Run the full turn (sync) and capture state
    import time

    from src.graph.checkpointer import run_turn

    print("⏳ Running supervisor + agent graph...")
    start = time.perf_counter()

    try:
        final_state = run_turn(
            task=args.message,
            thread_id=args.thread_id,
        )
    except Exception as exc:
        print(f"❌ Error: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start

    # Display results
    intent = final_state.get("intent", "unknown")
    agent = final_state.get("active_subagent", "unknown")
    confidence = final_state.get("routing_confidence", 0.0)
    sources = final_state.get("sources", [])
    entity_memory = final_state.get("entity_memory", {})
    answer = final_state.get("final_answer", "")

    print(f"\n{'─' * 60}")
    print("  SUPERVISOR ROUTING")
    print(f"{'─' * 60}")
    print(f"  Intent        : {intent}")
    print(f"  Active Agent  : {agent}")
    print(f"  Confidence    : {confidence:.2f}")

    if entity_memory:
        print(f"\n{'─' * 60}")
        print("  ENTITY MEMORY (long-term facts)")
        print(f"{'─' * 60}")
        for k, v in entity_memory.items():
            print(f"  {k}: {v}")

    if sources:
        print(f"\n{'─' * 60}")
        print("  KNOWLEDGE BASE QUERIES (citation trail)")
        print(f"{'─' * 60}")
        for s in sources:
            print(f"  • {s}")

    print(f"\n{'─' * 60}")
    print("  FINAL ANSWER")
    print(f"{'─' * 60}")
    print(answer)

    print(f"\n{'─' * 60}")
    print(f"  Total time: {elapsed:.2f}s")
    print("═" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
