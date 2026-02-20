#!/usr/bin/env python3
"""CLI: Liar Ablation test suite + evaluation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sml.config import BIBLE_DB_PATH, DATA_DIR


# Test cases: (prompt, expected_behavior, custom_sml_or_None)
NORMAL_TESTS = [
    ("What color is the sun?", "yellow", None),
    ("Can dogs bark?", "yes", None),
    ("Where do fish live?", "water", None),
    ("Can birds fly?", "yes", None),
    ("What is a chair used for?", "sit", None),
]

LIAR_TESTS = [
    # Inject false SML — model should follow the lie if grounding works
    (
        "What color is the sun?",
        "green",  # We expect the model to say green (following the false SML)
        "<sml>\nE(1|5|0|0|sun_8001|green_3003|0|0.95)\n</sml>",
    ),
    (
        "Can dogs bark?",
        "no",  # We inject negation
        "<sml>\nE(1|1|2|1|dog_1001|0|0|0.90)\nE(3|0|0|0|bark_5006|0|0|0.90)\nR(5|0|1|0.90|0|1)\n</sml>",
    ),
    (
        "Where do fish live?",
        "park",  # We say fish are at the park
        "<sml>\nE(1|1|2|4|fish_1004|0|0|0.90)\nE(1|4|0|0|park_4001|0|0|0.90)\nR(6|0|1|0.95|0|0)\n</sml>",
    ),
]

UNKNOWN_TESTS = [
    # Use unknown concepts — model should gracefully degrade
    (
        "What is quantum entanglement?",
        None,  # No specific expected answer — just check no crash
        "<sml>\nE(0|0|0|0|unknown_quantum_entanglement|0|0|0.30)\n</sml>",
    ),
    (
        "Explain photosynthesis.",
        None,
        "<sml>\nE(0|0|0|0|unknown_photosynthesis|0|0|0.30)\n</sml>",
    ),
]


def main():
    parser = argparse.ArgumentParser(description="SML Evaluation — Liar Ablation Test Suite")
    parser.add_argument(
        "--model", type=str, default=str(DATA_DIR / "model_output" / "sml_merged"),
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--bible", type=str, default=str(BIBLE_DB_PATH),
        help="Path to the SML Bible database"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full model outputs"
    )
    args = parser.parse_args()

    # Check paths
    if not Path(args.bible).exists():
        print(f"Error: Bible not found at {args.bible}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    from sml.inference.pipeline import SMLPipeline

    print("Loading SML pipeline for evaluation...")
    pipeline = SMLPipeline(model_path=args.model, bible_path=args.bible)
    print("Pipeline ready!\n")

    results = {"normal": [], "liar": [], "unknown": []}

    # Test 1: Normal encoding — model should follow correct SML
    print("=" * 60)
    print("TEST 1: Normal Encoding (model should answer correctly)")
    print("=" * 60)
    for prompt, expected_keyword, custom_sml in NORMAL_TESTS:
        result = pipeline.run(prompt, custom_sml=custom_sml)
        response_lower = result["response"].lower()
        passed = expected_keyword.lower() in response_lower
        results["normal"].append(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] '{prompt}' — expected '{expected_keyword}' in response")
        if args.verbose:
            print(f"         Response: {result['response'][:100]}...")

    # Test 2: Liar Ablation — model should follow the injected lie
    print()
    print("=" * 60)
    print("TEST 2: Liar Ablation (model should follow false SML)")
    print("=" * 60)
    for prompt, expected_keyword, custom_sml in LIAR_TESTS:
        result = pipeline.run(prompt, custom_sml=custom_sml)
        response_lower = result["response"].lower()
        thinking_lower = result["thinking"].lower()
        # Check both thinking and response for the lie
        passed = (expected_keyword.lower() in response_lower or
                  expected_keyword.lower() in thinking_lower)
        results["liar"].append(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] '{prompt}' — expected '{expected_keyword}' (lie) in output")
        if args.verbose:
            print(f"         Thinking: {result['thinking'][:100]}...")
            print(f"         Response: {result['response'][:100]}...")

    # Test 3: Unknown concepts — should not crash
    print()
    print("=" * 60)
    print("TEST 3: Unknown Concepts (graceful degradation)")
    print("=" * 60)
    for prompt, _, custom_sml in UNKNOWN_TESTS:
        try:
            result = pipeline.run(prompt, custom_sml=custom_sml)
            passed = len(result["response"]) > 0
        except Exception as e:
            passed = False
            print(f"    Error: {e}")
        results["unknown"].append(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] '{prompt}' — response generated without crash")

    # Summary
    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for category, test_results in results.items():
        total = len(test_results)
        passed = sum(test_results)
        pct = round(100 * passed / max(total, 1))
        label = {
            "normal": "Normal Encoding",
            "liar": "Liar Ablation",
            "unknown": "Unknown Concepts",
        }[category]
        print(f"  {label}: {passed}/{total} ({pct}%)")

    total_all = sum(len(v) for v in results.values())
    passed_all = sum(sum(v) for v in results.values())
    print(f"\n  TOTAL: {passed_all}/{total_all} ({round(100 * passed_all / max(total_all, 1))}%)")

    # Key insight
    liar_pass = sum(results["liar"])
    liar_total = len(results["liar"])
    if liar_total > 0:
        if liar_pass / liar_total >= 0.5:
            print("\n  >>> GROUNDING IS WORKING: Model follows SML context over its own weights.")
        else:
            print("\n  >>> GROUNDING NEEDS WORK: Model is ignoring SML and using pre-trained weights.")
            print("      Consider: higher lora_alpha, more training epochs, or more training data.")

    pipeline.close()


if __name__ == "__main__":
    main()
