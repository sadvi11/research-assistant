#!/usr/bin/env python3
"""Run the evaluation suite against the live pipeline.

    python scripts/evaluate.py                # full set
    python scripts/evaluate.py --category hallucination_bait
    python scripts/evaluate.py --dry-run      # scripted model, no API calls, $0

⚠️  Without --dry-run this calls the Anthropic API once per case (plus claim
    verification calls) and therefore costs money.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_config import configure_logging  # noqa: E402
from assistant import ResearchAssistant  # noqa: E402
from citations.validator import CitationValidator  # noqa: E402
from citations.verifier import ClaimVerifier  # noqa: E402
from config.settings import get_settings  # noqa: E402
from evaluation.dataset import EVAL_CASES, load_eval_corpus  # noqa: E402
from evaluation.runner import run_evaluation, save_report  # noqa: E402
from generation.client import ClaudeClient  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from retrieval.embeddings import get_embedder  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402
from retrieval.store import InMemoryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", help="run only one category")
    parser.add_argument("--dry-run", action="store_true",
                        help="use a scripted model; no API calls, no cost")
    parser.add_argument("--save", action="store_true", help="write a JSON report")
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = get_settings()

    embedder = get_embedder(settings, kind="hash" if args.dry_run else None)
    store = InMemoryStore()
    chunks = load_eval_corpus(IngestionPipeline(store=store, embedder=embedder, settings=settings))
    print(f"Evaluation corpus: {chunks} chunks\n")

    if args.dry_run:
        from evaluation.fake_model import ExtractiveFakeClient

        client = ExtractiveFakeClient(settings)
        print("Dry run: extractive stand-in model. No API calls, $0 billed.\n")
    else:
        if not ClaudeClient.credentials_available():
            print("error: no Anthropic credentials found.\n"
                  "       Set ANTHROPIC_API_KEY, run `ant auth login`, or use --dry-run.",
                  file=sys.stderr)
            return 2
        client = ClaudeClient(settings)

    assistant = ResearchAssistant(
        HybridRetriever(store, embedder, settings=settings), client, settings=settings,
        validator=CitationValidator(), verifier=ClaimVerifier(client, settings=settings),
    )

    cases = [c for c in EVAL_CASES if not args.category or c.category == args.category]
    if not cases:
        print(f"error: no cases in category {args.category!r}", file=sys.stderr)
        return 2

    summary = run_evaluation(assistant, cases)
    print()
    print(summary.report())

    if args.save:
        print(f"\nReport written to {save_report(summary)}")

    # A false answer - answering when the evidence does not support it - is the
    # failure this project exists to prevent, so it fails the run.
    return 1 if summary.false_answer_rate > 0 or summary.hallucination_incidents else 0


if __name__ == "__main__":
    raise SystemExit(main())
