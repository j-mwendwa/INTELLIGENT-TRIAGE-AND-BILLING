#!/usr/bin/env python3
"""
scripts/ingest.py — CLI ingestion script.

Usage:
    python scripts/ingest.py --dir data/raw
    python scripts/ingest.py --dir data/raw/billing --collection billing_kb
    python scripts/ingest.py --dir data/raw/technical --collection technical_kb
    python scripts/ingest.py --dir data/raw/compliance --collection compliance_kb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Intelligent Triage & Billing vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py --dir data/raw
  python scripts/ingest.py --dir data/raw/billing --collection billing_kb
  python scripts/ingest.py --dir data/raw/technical --collection technical_kb
  python scripts/ingest.py --dir data/raw/compliance --collection compliance_kb
        """,
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--collection",
        default="knowledge_base",
        help="Target vector store collection name (default: knowledge_base)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Setup
    from src.config import settings
    from src.core.logging import setup_logging

    log_level = "DEBUG" if args.verbose else settings.log_level
    setup_logging(log_level=log_level, app_env=settings.app_env)

    import structlog

    logger = structlog.get_logger("ingest")

    directory = Path(args.dir)
    if not directory.exists():
        print(f"❌ Directory not found: {directory}", file=sys.stderr)
        return 1

    print("\n🔄 Setting up LlamaIndex embeddings...")
    from src.core.llamaindex_setup import setup_llamaindex

    setup_llamaindex()

    print(f"📂 Ingesting: {directory.resolve()}")
    print(f"📦 Collection: {args.collection}")
    print()

    start = time.perf_counter()
    try:
        from src.ingestion.llamaindex_pipeline import ingest_directory

        count = ingest_directory(directory=directory, collection=args.collection)
        elapsed = time.perf_counter() - start

        print("✅ Ingestion complete!")
        print(f"   Documents indexed : {count}")
        print(f"   Collection        : {args.collection}")
        print(f"   Time elapsed      : {elapsed:.1f}s")
        print()
        return 0

    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"❌ Ingestion failed after {elapsed:.1f}s: {exc}", file=sys.stderr)
        logger.error("ingestion_failed", error=str(exc), directory=str(directory), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
