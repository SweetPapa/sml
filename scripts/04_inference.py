#!/usr/bin/env python3
"""CLI: Interactive SML inference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sml.config import BIBLE_DB_PATH, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Interactive SML inference")
    parser.add_argument(
        "--model", type=str, default=str(DATA_DIR / "model_output" / "sml_merged"),
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--bible", type=str, default=str(BIBLE_DB_PATH),
        help="Path to the SML Bible database"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query to run (non-interactive mode)"
    )
    args = parser.parse_args()

    # Check paths
    if not Path(args.bible).exists():
        print(f"Error: Bible not found at {args.bible}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Run scripts/03_train.py first (with --merge flag).")
        sys.exit(1)

    from sml.inference.pipeline import SMLPipeline

    print("Loading SML pipeline...")
    pipeline = SMLPipeline(model_path=args.model, bible_path=args.bible)
    print("Pipeline ready!\n")

    if args.query:
        # Single query mode
        _run_query(pipeline, args.query)
    else:
        # Interactive mode
        print("SML Interactive Inference")
        print("Type your question (or 'quit' to exit)")
        print("-" * 40)
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            _run_query(pipeline, user_input)

    pipeline.close()
    print("\nGoodbye!")


def _run_query(pipeline, query: str):
    """Run a single query through the pipeline and display results."""
    result = pipeline.run(query)

    print("\n--- SML Block ---")
    print(result["sml_block"])
    print("\n--- Thinking ---")
    print(result["thinking"] or "(no thinking block)")
    print("\n--- Response ---")
    print(result["response"])


if __name__ == "__main__":
    main()
