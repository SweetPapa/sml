#!/usr/bin/env python3
"""CLI: Build the SML Bible (micro or full mode)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sml.config import BIBLE_DB_PATH


def main():
    parser = argparse.ArgumentParser(description="Build the SML Bible database")
    parser.add_argument(
        "--mode", choices=["micro", "full"], default="micro",
        help="Build mode: 'micro' for ~50 hand-crafted concepts, 'full' for ConceptNet+WordNet"
    )
    parser.add_argument(
        "--output", type=str, default=str(BIBLE_DB_PATH),
        help="Output path for the SQLite database"
    )
    parser.add_argument(
        "--max-concepts", type=int, default=100000,
        help="Max concepts to import in full mode"
    )
    parser.add_argument(
        "--conceptnet-cache", type=str, default=None,
        help="Path to cached ConceptNet CSV.gz file (full mode only)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Building SML Bible ({args.mode} mode)")
    print("=" * 60)

    # Remove existing DB if present
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()
        print(f"Removed existing database: {args.output}")

    if args.mode == "micro":
        from sml.bible.micro_builder import build_micro_bible
        build_micro_bible(args.output)
    else:
        from sml.bible.builder import build_full_bible
        build_full_bible(
            args.output,
            conceptnet_cache=args.conceptnet_cache,
            max_concepts=args.max_concepts,
        )

    print("\nDone! Bible database created at:", args.output)


if __name__ == "__main__":
    main()
