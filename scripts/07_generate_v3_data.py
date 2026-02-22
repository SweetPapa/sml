#!/usr/bin/env python3
"""CLI: Generate V3 SML training data with algorithmic CoT."""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from sml.config import (
    BIBLE_DB_PATH,
    V3_CATEGORY_DISTRIBUTION,
    V3_MANIFEST_PATH,
    V3_TRAINING_DATA_PATH,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate V3 SML training data with algorithmic CoT"
    )
    parser.add_argument(
        "--bible", type=str, default=str(BIBLE_DB_PATH),
        help="Path to the SML Bible database",
    )
    parser.add_argument(
        "--output", type=str, default=str(V3_TRAINING_DATA_PATH),
        help="Output path for JSONL training data",
    )
    parser.add_argument(
        "--manifest", type=str, default=str(V3_MANIFEST_PATH),
        help="Output path for manifest JSON",
    )
    parser.add_argument(
        "--groq-api-key", type=str, default=None,
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--num-examples", type=int, default=None,
        help="Override total example count (scales category distribution proportionally)",
    )
    parser.add_argument(
        "--category", choices=["A", "B", "C", "D"], default=None,
        help="Generate a single category only",
    )
    parser.add_argument(
        "--mode", choices=["test", "full"], default=None,
        help="Preset: 'test' (50 examples) or 'full' (500 examples)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per LLM step",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation after generation",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Determine category distribution
    if args.mode == "test":
        total = 50
    elif args.mode == "full":
        total = 500
    elif args.num_examples is not None:
        total = args.num_examples
    else:
        total = sum(V3_CATEGORY_DISTRIBUTION.values())

    if args.category:
        # Single category mode
        category_counts = {args.category: total}
    elif args.num_examples is not None or args.mode:
        # Scale the default distribution proportionally
        default_total = sum(V3_CATEGORY_DISTRIBUTION.values())
        scale = total / default_total
        category_counts = {}
        remaining = total
        cats = list(V3_CATEGORY_DISTRIBUTION.keys())
        for cat in cats[:-1]:
            n = max(1, round(V3_CATEGORY_DISTRIBUTION[cat] * scale))
            category_counts[cat] = n
            remaining -= n
        category_counts[cats[-1]] = max(1, remaining)
    else:
        category_counts = dict(V3_CATEGORY_DISTRIBUTION)

    # Check Bible exists
    if not Path(args.bible).exists():
        print(f"Error: Bible database not found at {args.bible}")
        print("Run scripts/01_build_bible.py first.")
        sys.exit(1)

    # Get API key
    api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key required. Set GROQ_API_KEY env var or use --groq-api-key")
        sys.exit(1)

    print("=" * 60)
    print("SML V3 Training Data Generation")
    print("=" * 60)
    print(f"Bible: {args.bible}")
    print(f"Output: {args.output}")
    print(f"Manifest: {args.manifest}")
    print(f"Category distribution: {category_counts}")
    print(f"Total examples: {sum(category_counts.values())}")
    print(f"Seed: {args.seed}")
    print(f"Max retries/step: {args.max_retries}")

    from sml.training.data_generator_v3 import generate_v3_training_data

    output_path = generate_v3_training_data(
        bible_path=args.bible,
        groq_api_key=api_key,
        output_path=args.output,
        manifest_path=args.manifest,
        category_counts=category_counts,
        seed=args.seed,
        max_retries_per_step=args.max_retries,
        validate=args.validate,
    )

    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
