#!/usr/bin/env python3
"""CLI: Validate Bible concepts against curated ground truth."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sml.config import BIBLE_DB_PATH

# Curated ground truth: concept -> expected HasProperty targets
GROUND_TRUTH = {
    "sun": ["yellow", "hot", "bright"],
    "sky": ["blue"],
    "grass": ["green", "soft"],
    "fire": ["red", "hot", "bright", "dangerous"],
    "snow": ["white", "cold", "soft"],
    "ice": ["cold", "hard"],
    "ocean": ["blue", "deep", "salty", "big"],
    "night": ["dark", "quiet"],
    "tree": ["green", "big", "tall"],
    "elephant": ["big", "heavy", "slow"],
    "mouse": ["small", "light", "quiet", "fast"],
    "apple": ["red", "sweet", "round"],
    "water": ["cold", "clear"],
    "milk": ["white", "healthy"],
    "dog": ["brown", "fast", "friendly", "loyal"],
    "cat": ["small", "independent", "quick"],
    "penguin": ["black", "white", "cute"],
    "snake": ["long"],
    "bird": ["light"],
    "bread": ["soft"],
    "ball": ["round"],
    "book": ["useful"],
    "park": ["green"],
}

# Curated capability ground truth: concept -> expected CapableOf targets
CAPABILITY_TRUTH = {
    "dog": ["bark", "run", "swim", "chase", "hear", "eat", "wag"],
    "cat": ["purr", "run", "climb", "meow", "chase", "walk", "eat"],
    "bird": ["fly", "sing", "build", "walk"],
    "fish": ["swim", "eat"],
    "penguin": ["swim", "walk", "eat"],
    "snake": ["swim", "eat"],
    "elephant": ["walk", "swim", "eat", "drink"],
    "mouse": ["run", "climb", "eat", "swim"],
    "person": ["read", "run", "walk", "eat", "drink", "cook"],
    "child": ["play", "learn", "read"],
}

# Taxonomy ground truth: concept -> expected IsA targets
TAXONOMY_TRUTH = {
    "dog": ["animal", "mammal", "pet"],
    "cat": ["animal", "mammal", "pet"],
    "bird": ["animal"],
    "fish": ["animal"],
    "penguin": ["bird", "animal"],
    "snake": ["reptile", "animal"],
    "elephant": ["mammal", "animal"],
    "mouse": ["mammal", "animal"],
    "sun": ["star"],
    "apple": ["fruit"],
    "grass": ["plant"],
    "tree": ["plant"],
}

# Body part ground truth: concept -> expected HasA targets
BODY_PART_TRUTH = {
    "dog": ["leg", "tail", "ear", "fur"],
    "cat": ["leg", "tail", "fur"],
    "bird": ["wing", "leg"],
    "fish": ["scale"],
    "penguin": ["wing"],
    "snake": ["scale"],
    "elephant": ["leg", "trunk", "ear"],
    "mouse": ["tail"],
    "tree": ["leaf"],
}

# Antonym ground truth: concept -> expected Antonym targets
ANTONYM_TRUTH = {
    "hot": ["cold"],
    "cold": ["hot"],
    "big": ["small"],
    "small": ["big"],
    "fast": ["slow"],
    "slow": ["fast"],
    "dark": ["bright"],
    "bright": ["dark"],
    "old": ["young"],
    "heavy": ["light"],
    "light": ["heavy"],
    "soft": ["hard"],
    "hard": ["soft"],
    "quiet": ["loud"],
    "loud": ["quiet"],
}


def _validate_relation_type(bible, truth_dict, relation_type_id, relation_name):
    """Validate a set of ground truth entries for a given relation type."""
    results = []
    for concept_text, expected_targets in truth_dict.items():
        concept = bible.lookup_concept(concept_text)
        if concept is None:
            results.append({
                "concept": concept_text,
                "type": relation_name,
                "status": "MISSING_CONCEPT",
                "expected": expected_targets,
                "found": [],
                "missing": expected_targets,
            })
            continue

        rels = bible.get_outgoing_relations(concept["id"])
        type_rels = [r for r in rels if r["relation_type_id"] == relation_type_id]
        found_targets = [r["target_text"] for r in type_rels]

        missing = [t for t in expected_targets if t not in found_targets]
        status = "PASS" if not missing else "MISSING"

        results.append({
            "concept": concept_text,
            "type": relation_name,
            "status": status,
            "expected": expected_targets,
            "found": found_targets,
            "missing": missing,
        })
    return results


def validate_bible(bible_path: str, verbose: bool = False) -> dict:
    """Validate Bible against ground truth.

    Returns dict with per-entry results and summary.
    """
    from sml.bible.query import Bible

    bible = Bible(bible_path)
    results = []

    # Validate HasProperty relations (type_id=4)
    results.extend(_validate_relation_type(bible, GROUND_TRUTH, 4, "HasProperty"))

    # Validate CapableOf relations (type_id=5)
    results.extend(_validate_relation_type(bible, CAPABILITY_TRUTH, 5, "CapableOf"))

    # Validate IsA relations (type_id=1)
    results.extend(_validate_relation_type(bible, TAXONOMY_TRUTH, 1, "IsA"))

    # Validate HasA relations (type_id=3)
    results.extend(_validate_relation_type(bible, BODY_PART_TRUTH, 3, "HasA"))

    # Validate Antonym relations (type_id=22)
    results.extend(_validate_relation_type(bible, ANTONYM_TRUTH, 22, "Antonym"))

    bible.close()

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    missing = sum(1 for r in results if r["status"] == "MISSING")
    missing_concept = sum(1 for r in results if r["status"] == "MISSING_CONCEPT")

    summary = {
        "total": total,
        "pass": passed,
        "missing": missing,
        "missing_concept": missing_concept,
        "pass_pct": round(100 * passed / max(total, 1), 1),
        "results": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate SML Bible against ground truth")
    parser.add_argument(
        "--bible", type=str, default=str(BIBLE_DB_PATH),
        help="Path to the SML Bible database"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-entry results"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Output JSON summary to this path"
    )
    args = parser.parse_args()

    if not Path(args.bible).exists():
        print(f"Error: Bible not found at {args.bible}")
        print("Run scripts/01_build_bible.py --mode micro first.")
        sys.exit(1)

    print("=" * 60)
    print("SML Bible Validation")
    print("=" * 60)

    summary = validate_bible(args.bible, verbose=args.verbose)

    # Print results
    for entry in summary["results"]:
        status = entry["status"]
        icon = {"PASS": "PASS", "MISSING": "MISS", "MISSING_CONCEPT": "GONE"}[status]
        print(f"  [{icon}] {entry['concept']} ({entry['type']}): "
              f"expected {entry['expected']}, found {entry['found']}"
              + (f", missing {entry.get('missing', [])}" if entry.get("missing") else ""))

    print()
    print(f"Summary: {summary['pass']}/{summary['total']} PASS "
          f"({summary['pass_pct']}%)")
    if summary["missing"] > 0:
        print(f"  {summary['missing']} entries with missing relations")
    if summary["missing_concept"] > 0:
        print(f"  {summary['missing_concept']} concepts not found in Bible")

    # Write JSON output
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nJSON summary written to {args.json}")


if __name__ == "__main__":
    main()
