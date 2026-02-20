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
    args = parser.parse_args()

    # Get API key
    api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key required. Set GROQ_API_KEY env var or use --groq-api-key")
        sys.exit(1)

    # Check Bible exists
    if not Path(args.bible).exists():
        print(f"Error: Bible database not found at {args.bible}")
        print("Run scripts/01_build_bible.py first.")
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
