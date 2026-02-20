#!/usr/bin/env python3
"""CLI: Generate SML training data using the inverted pipeline."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sml.config import BIBLE_DB_PATH, TRAINING_DATA_PATH


def main():
    parser = argparse.ArgumentParser(description="Generate SML training data via Groq")
    parser.add_argument(
        "--num-examples", type=int, default=200,
        help="Number of training examples to generate"
    )
    parser.add_argument(
        "--bible", type=str, default=str(BIBLE_DB_PATH),
        help="Path to the SML Bible database"
    )
    parser.add_argument(
        "--output", type=str, default=str(TRAINING_DATA_PATH),
        help="Output path for JSONL training data"
    )
    parser.add_argument(
        "--groq-api-key", type=str, default=None,
        help="Groq API key (or set GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation after generation"
    )
    parser.add_argument(
        "--coverage-only", action="store_true",
        help="Only compute encoder coverage stats (no Groq API calls)"
    )
    args = parser.parse_args()

    # Check Bible exists
    if not Path(args.bible).exists():
        print(f"Error: Bible database not found at {args.bible}")
        print("Run scripts/01_build_bible.py first.")
        sys.exit(1)

    # Coverage-only mode
    if args.coverage_only:
        print("=" * 60)
        print("SML Encoder Coverage Report")
        print("=" * 60)
        from sml.training.data_generator import compute_coverage, MICRO_PROMPTS
        stats = compute_coverage(bible_path=args.bible)
        print(f"Total prompts: {stats['total_prompts']}")
        print(f"Prompts with known entities: {stats['prompts_with_entities']}")
        print(f"Prompts with relations: {stats['prompts_with_relations']}")
        print(f"Known concepts: {stats['known_concepts']}")
        print(f"Unknown concepts: {stats['unknown_concepts']}")
        print(f"Avg entities/block: {stats['avg_entities_per_block']}")
        print(f"Avg relations/block: {stats['avg_relations_per_block']}")
        print(f"Coverage: {stats['coverage_pct']}%")
        if stats['empty_blocks'] > 0:
            print(f"Empty blocks: {stats['empty_blocks']}")
        # Show worst prompts (most unknowns)
        worst = sorted(stats["per_prompt"], key=lambda p: p["unknown"], reverse=True)[:10]
        if worst and worst[0]["unknown"] > 0:
            print(f"\nTop unknowns:")
            for p in worst:
                if p["unknown"] > 0:
                    print(f"  {p['unknown']} unknown: {p['prompt']}")
        print("\nDone!")
        sys.exit(0)

    # Get API key
    api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key required. Set GROQ_API_KEY env var or use --groq-api-key")
        sys.exit(1)

    print("=" * 60)
    print("SML Training Data Generation")
    print("=" * 60)
    print(f"Bible: {args.bible}")
    print(f"Output: {args.output}")
    print(f"Examples: {args.num_examples}")
    print()

    from sml.training.data_generator import generate_training_data

    output_path = generate_training_data(
        bible_path=args.bible,
        groq_api_key=api_key,
        output_path=args.output,
        num_examples=args.num_examples,
    )

    if args.validate:
        print("\nRunning validation...")
        from sml.training.validator import validate_training_data
        validate_training_data(output_path, args.bible)

    print("\nDone!")


if __name__ == "__main__":
    main()
