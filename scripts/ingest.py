#!/usr/bin/env python3
"""Ingest documents into the corpus.

    python scripts/ingest.py data/corpus --url-prefix https://docs.aws.amazon.com
    python scripts/ingest.py path/to/guide.pdf --url https://example.gov/guide.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.deps import get_ingestion  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from config.sources import SourceTier, get_trust_policy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="file or directory to ingest")
    parser.add_argument("--url", help="canonical source URL (decides the trust tier)")
    parser.add_argument("--url-prefix", help="URL prefix when ingesting a directory")
    parser.add_argument("--min-tier", type=int, choices=[1, 2, 3, 4], default=4,
                        help="reject sources below this tier (1 = official only)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    configure_logging("WARNING" if args.quiet else "INFO")
    path = Path(args.path)
    if not path.exists():
        print(f"error: no such path: {path}", file=sys.stderr)
        return 2

    if args.url:
        tier = get_trust_policy().tier_for_url(args.url)
        print(f"Source tier: {tier.name} ({tier.label})")
        if tier > SourceTier(args.min_tier):
            print(f"error: {args.url} is tier {tier.name}, below the required "
                  f"{SourceTier(args.min_tier).name}", file=sys.stderr)
            return 1

    pipeline = get_ingestion()
    result = (
        pipeline.ingest_directory(path, url_prefix=args.url_prefix)
        if path.is_dir()
        else pipeline.ingest_file(path, url=args.url)
    )
    print(f"Ingested: {result.summary()}")
    for source, reason in result.skipped:
        print(f"  skipped {source}: {reason}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
