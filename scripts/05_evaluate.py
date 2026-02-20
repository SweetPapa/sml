#!/usr/bin/env python3
"""CLI: Expanded evaluation suite — 50 tests across 5 categories."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sml.config import BIBLE_DB_PATH, DATA_DIR


# ── Test definitions ──────────────────────────────────────────────────────────
# Format: (prompt, keywords, anti_keywords, custom_sml_or_None)
#   keywords: list of acceptable keywords (any match → PASS, partial synonyms → PARTIAL)
#   anti_keywords: list of contradictions (any match → FAIL regardless)

NORMAL_TESTS = [
    ("What color is the sun?", ["yellow"], ["blue", "green", "red", "white"], None),
    ("Can dogs bark?", ["yes", "bark", "can"], ["no", "cannot", "can't"], None),
    ("Where do fish live?", ["water", "ocean"], ["park", "house", "school"], None),
    ("Can birds fly?", ["yes", "fly", "can"], ["no", "cannot", "can't"], None),
    ("What is a chair used for?", ["sit", "sitting", "seat"], ["eat", "run"], None),
    ("What color is the sky?", ["blue"], ["red", "green", "yellow"], None),
    ("What color is grass?", ["green"], ["blue", "red", "yellow"], None),
    ("Is fire hot?", ["yes", "hot"], ["cold", "no"], None),
    ("Is snow cold?", ["yes", "cold"], ["hot", "warm", "no"], None),
    ("Is an elephant big?", ["yes", "big", "large"], ["small", "tiny", "no"], None),
    ("Is a mouse small?", ["yes", "small", "little", "tiny"], ["big", "large", "no"], None),
    ("Where do children go during the day?", ["school"], ["park", "kitchen"], None),
    ("What color is fire?", ["red", "orange"], ["blue", "green"], None),
    ("What is ice made of?", ["water"], ["fire", "air"], None),
    ("Is the sun bright?", ["yes", "bright"], ["dark", "dim", "no"], None),
]

LIAR_TESTS = [
    # Inject false SML — model should follow the lie if grounding works
    (
        "What color is the sun?",
        ["green"],
        ["yellow"],
        "<sml>\nE(1|5|0|0|sun_8001|green_3003|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "What color is the sky?",
        ["red"],
        ["blue"],
        "<sml>\nE(1|5|0|0|sky_8002|red_3001|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "What color is grass?",
        ["yellow"],
        ["green"],
        "<sml>\nE(1|1|3|0|grass_9001|yellow_3004|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "Where do fish live?",
        ["park"],
        ["water", "ocean"],
        "<sml>\nE(1|1|2|4|fish_1004|0|0|0.90)\nE(1|4|0|0|park_4001|0|0|0.90)\nR(AtLocation|0|1|0.95|0|0)\n</sml>",
    ),
    (
        "Is the sun hot?",
        ["cold", "no"],
        ["hot", "yes"],
        "<sml>\nE(1|5|0|0|sun_8001|cold_3202|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "Is an elephant big?",
        ["small", "no"],
        ["big", "large", "yes"],
        "<sml>\nE(1|1|2|3|elephant_1007|small_3102|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "Can birds fly?",
        ["no", "cannot", "can't"],
        ["yes", "can fly"],
        "<sml>\nE(1|1|2|2|bird_1003|0|0|0.90)\nE(3|0|0|0|fly_5004|0|0|0.90)\nR(NOT_CapableOf|0|1|0.90|0|0)\n</sml>",
    ),
    (
        "Can dogs bark?",
        ["no", "cannot", "can't"],
        ["yes"],
        "<sml>\nE(1|1|2|1|dog_1001|0|0|0.90)\nE(3|0|0|0|bark_5006|0|0|0.90)\nR(NOT_CapableOf|0|1|0.90|0|0)\n</sml>",
    ),
    (
        "Is snow white?",
        ["black", "no"],
        ["white", "yes"],
        "<sml>\nE(1|5|0|0|snow_8005|black_3007|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "Where do dogs like to go?",
        ["school", "kitchen"],
        ["park"],
        "<sml>\nE(1|1|2|1|dog_1001|0|0|0.90)\nE(1|4|0|0|school_4003|0|0|0.90)\nR(AtLocation|0|1|0.85|0|0)\n</sml>",
    ),
    (
        "Is fire hot?",
        ["cold", "no"],
        ["hot", "warm", "yes"],
        "<sml>\nE(1|5|0|0|fire_8004|cold_3202|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "What color is milk?",
        ["red"],
        ["white"],
        "<sml>\nE(1|3|0|0|milk_6004|red_3001|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "Is a mouse small?",
        ["big", "large", "no"],
        ["small", "tiny", "yes"],
        "<sml>\nE(1|1|2|3|mouse_1008|big_3101|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "Can fish swim?",
        ["no", "cannot"],
        ["yes", "can swim"],
        "<sml>\nE(1|1|2|4|fish_1004|0|0|0.90)\nE(3|0|0|0|swim_5005|0|0|0.90)\nR(NOT_CapableOf|0|1|0.90|0|0)\n</sml>",
    ),
    (
        "Is ice cold?",
        ["hot", "warm", "no"],
        ["cold", "yes"],
        "<sml>\nE(1|3|0|0|ice_6005|hot_3201|0|0.95)\nR(HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
]

NEGATION_TESTS = [
    # Tests where the SML correctly encodes negation via NOT_ prefix
    (
        "Can penguins fly?",
        ["no", "cannot", "can't", "unable"],
        ["yes", "can fly"],
        "<sml>\nE(1|1|2|3|penguin_1005|0|0|0.90)\nE(3|0|0|0|fly_5004|0|0|0.90)\nR(NOT_CapableOf|0|1|0.95|0|0)\n</sml>",
    ),
    (
        "Can fish walk?",
        ["no", "cannot", "can't"],
        ["yes", "can walk"],
        "<sml>\nE(1|1|2|4|fish_1004|0|0|0.90)\nE(3|0|0|0|walk_5010|0|0|0.90)\nR(NOT_CapableOf|0|1|0.95|0|0)\n</sml>",
    ),
    (
        "Can snakes hear?",
        ["no", "cannot", "can't"],
        ["yes"],
        "<sml>\nE(1|1|2|3|snake_1006|0|0|0.90)\nE(3|0|0|0|hear_5011|0|0|0.90)\nR(NOT_CapableOf|0|1|0.95|0|0)\n</sml>",
    ),
    (
        "Is ice hot?",
        ["no", "not hot", "cold"],
        ["yes", "hot"],
        "<sml>\nE(1|3|0|0|ice_6005|cold_3202|0|0.90)\nR(NOT_HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "Is the night bright?",
        ["no", "dark", "not bright"],
        ["yes", "bright"],
        "<sml>\nE(1|5|0|0|night_8003|dark_3302|0|0.90)\nR(NOT_HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "Can elephants fly?",
        ["no", "cannot", "can't"],
        ["yes"],
        "<sml>\nE(1|1|2|3|elephant_1007|0|0|0.90)\nE(3|0|0|0|fly_5004|0|0|0.90)\nR(NOT_CapableOf|0|1|0.95|0|0)\n</sml>",
    ),
    (
        "Can mice fly?",
        ["no", "cannot", "can't"],
        ["yes"],
        "<sml>\nE(1|1|2|3|mouse_1008|0|0|0.90)\nE(3|0|0|0|fly_5004|0|0|0.90)\nR(NOT_CapableOf|0|1|0.95|0|0)\n</sml>",
    ),
    (
        "Is snow hot?",
        ["no", "cold", "not hot"],
        ["yes", "hot"],
        "<sml>\nE(1|5|0|0|snow_8005|cold_3202|0|0.90)\nR(NOT_HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
    (
        "Can fish fly?",
        ["no", "cannot", "can't"],
        ["yes"],
        "<sml>\nE(1|1|2|4|fish_1004|0|0|0.90)\nE(3|0|0|0|fly_5004|0|0|0.90)\nR(NOT_CapableOf|0|1|0.95|0|0)\n</sml>",
    ),
    (
        "Is the ocean hot?",
        ["no", "cold", "not hot"],
        ["yes", "hot"],
        "<sml>\nE(1|4|0|0|ocean_4004|cold_3202|0|0.90)\nR(NOT_HasProperty|0|0|0.95|0|0)\n</sml>",
    ),
]

UNKNOWN_TESTS = [
    (
        "What is quantum entanglement?",
        None,  # No specific expected answer — just check no crash and non-empty response
        None,
        "<sml>\nE(0|0|0|0|unknown_quantum_entanglement|0|0|0.30)\n</sml>",
    ),
    (
        "Explain photosynthesis.",
        None,
        None,
        "<sml>\nE(0|0|0|0|unknown_photosynthesis|0|0|0.30)\n</sml>",
    ),
    (
        "What is the meaning of democracy?",
        None,
        None,
        "<sml>\nE(0|0|0|0|unknown_democracy|0|0|0.30)\n</sml>",
    ),
    (
        "Describe blockchain technology.",
        None,
        None,
        "<sml>\nE(0|0|0|0|unknown_blockchain|0|0|0.30)\n</sml>",
    ),
    (
        "What causes gravity?",
        None,
        None,
        "<sml>\nE(0|0|0|0|unknown_gravity|0|0|0.30)\n</sml>",
    ),
]

MULTI_ENTITY_TESTS = [
    (
        "The big brown dog ran in the park. Describe the scene.",
        ["dog", "park"],
        [],
        None,
    ),
    (
        "The small black cat is sleeping on the chair in the kitchen. What is happening?",
        ["cat", "sleep", "chair"],
        [],
        None,
    ),
    (
        "A child is reading a book at school. Describe the scene.",
        ["child", "book", "school"],
        [],
        None,
    ),
    (
        "The red ball is in the park near the big tree. What do you see?",
        ["ball", "park", "tree"],
        [],
        None,
    ),
    (
        "A dog and a cat are playing in the house. What animals are there?",
        ["dog", "cat"],
        [],
        None,
    ),
]


def score_response(
    response: str,
    thinking: str,
    keywords: list[str] | None,
    anti_keywords: list[str] | None,
) -> str:
    """Score a response as PASS, PARTIAL, or FAIL.

    - PASS: primary keyword found, no contradictions
    - PARTIAL: synonym/related found, or keyword in thinking but not response
    - FAIL: contradiction found, or no keywords matched at all
    """
    if keywords is None:
        # Unknown concept test — just check non-empty response
        return "PASS" if len(response) > 0 else "FAIL"

    combined = (response + " " + thinking).lower()
    response_lower = response.lower()

    # Check for contradictions first
    if anti_keywords:
        for anti in anti_keywords:
            if anti.lower() in response_lower:
                return "FAIL"

    # Check for primary keyword match in response
    for kw in keywords:
        if kw.lower() in response_lower:
            return "PASS"

    # Check if keyword appears in thinking but not response (partial)
    for kw in keywords:
        if kw.lower() in combined:
            return "PARTIAL"

    return "FAIL"


def run_test_category(
    pipeline,
    category_name: str,
    tests: list,
    verbose: bool = False,
) -> dict:
    """Run a test category and return results."""
    results = {"pass": 0, "partial": 0, "fail": 0, "details": []}

    for test in tests:
        prompt, keywords, anti_keywords, custom_sml = test

        try:
            result = pipeline.run(prompt, custom_sml=custom_sml)
            grade = score_response(
                result["response"], result["thinking"],
                keywords, anti_keywords,
            )
        except Exception as e:
            grade = "FAIL"
            result = {"response": f"ERROR: {e}", "thinking": "", "sml_block": ""}

        results[grade.lower()] += 1
        detail = {
            "prompt": prompt,
            "grade": grade,
            "expected_keywords": keywords,
            "response_preview": result["response"][:150],
        }
        results["details"].append(detail)

        icon = {"PASS": "PASS", "PARTIAL": "PART", "FAIL": "FAIL"}[grade]
        kw_str = str(keywords[:3]) if keywords else "N/A"
        print(f"  [{icon}] {prompt[:60]:60s} expect={kw_str}")
        if verbose:
            print(f"         Response: {result['response'][:100]}...")
            if result.get("thinking"):
                print(f"         Thinking: {result['thinking'][:100]}...")

    return results


def main():
    parser = argparse.ArgumentParser(description="SML Evaluation — 50-Test Suite")
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
    parser.add_argument(
        "--output-dir", type=str, default=str(DATA_DIR / "eval_results"),
        help="Directory for JSON output"
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

    all_results = {}

    categories = [
        ("normal_encoding", "Normal Encoding (correct SML → correct answer)", NORMAL_TESTS),
        ("liar_ablation", "Liar Ablation (false SML → model follows the lie)", LIAR_TESTS),
        ("negation", "Negation (NOT_ relations → model says no/cannot)", NEGATION_TESTS),
        ("unknown", "Unknown Concepts (graceful degradation)", UNKNOWN_TESTS),
        ("multi_entity", "Multi-Entity (3+ entities, multiple relations)", MULTI_ENTITY_TESTS),
    ]

    for cat_key, cat_label, tests in categories:
        print("=" * 70)
        print(f"  {cat_label} ({len(tests)} tests)")
        print("=" * 70)
        all_results[cat_key] = run_test_category(
            pipeline, cat_key, tests, verbose=args.verbose
        )
        print()

    # Summary
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total_pass = 0
    total_partial = 0
    total_fail = 0
    total_tests = 0

    for cat_key, cat_label, tests in categories:
        r = all_results[cat_key]
        cat_total = r["pass"] + r["partial"] + r["fail"]
        total_pass += r["pass"]
        total_partial += r["partial"]
        total_fail += r["fail"]
        total_tests += cat_total
        pct = round(100 * r["pass"] / max(cat_total, 1))
        label = cat_label.split(" (")[0]
        print(f"  {label:25s}: {r['pass']} PASS / {r['partial']} PARTIAL / {r['fail']} FAIL  ({pct}%)")

    overall_pct = round(100 * total_pass / max(total_tests, 1))
    print(f"\n  {'TOTAL':25s}: {total_pass} PASS / {total_partial} PARTIAL / {total_fail} FAIL  ({overall_pct}%)")

    # Grounding insight
    liar = all_results.get("liar_ablation", {})
    liar_total = liar.get("pass", 0) + liar.get("partial", 0) + liar.get("fail", 0)
    if liar_total > 0:
        liar_pct = liar["pass"] / liar_total
        if liar_pct >= 0.5:
            print("\n  >>> GROUNDING IS WORKING: Model follows SML context over its own weights.")
        else:
            print("\n  >>> GROUNDING NEEDS WORK: Model is ignoring SML and using pre-trained weights.")

    # Save JSON output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Strip details for summary output
    summary_results = {}
    for k, v in all_results.items():
        summary_results[k] = {
            "pass": v["pass"],
            "partial": v["partial"],
            "fail": v["fail"],
        }

    output_data = {
        "run_id": run_id,
        "model": str(args.model),
        "total_tests": total_tests,
        "total_pass": total_pass,
        "total_partial": total_partial,
        "total_fail": total_fail,
        "overall_pct": overall_pct,
        "results": summary_results,
        "details": {k: v["details"] for k, v in all_results.items()},
    }

    output_path = output_dir / f"{run_id}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    pipeline.close()


if __name__ == "__main__":
    main()
