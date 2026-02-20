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
    "sun": ["yellow", "hot"],
    "sky": ["blue"],
    "grass": ["green"],
    "fire": ["red", "hot"],
    "snow": ["white", "cold"],
    "ice": ["cold"],
    "ocean": ["blue"],
    "night": ["dark"],
    "tree": ["green"],
    "elephant": ["big", "heavy"],
    "mouse": ["small"],
    "apple": ["red"],
    "water": ["cold"],
    "milk": ["white"],
    "dog": ["brown", "fast"],
    "cat": ["small"],
    "penguin": ["black", "white"],
}

# Curated capability ground truth: concept -> expected CapableOf targets
CAPABILITY_TRUTH = {
    "dog": ["bark", "run", "swim"],
    "bird": ["fly"],
    "fish": ["swim"],
    "penguin": ["swim", "walk"],
    "snake": ["swim"],
    "elephant": ["walk"],
}


def validate_bible(bible_path: str, verbose: bool = False) -> dict:
    """Validate Bible against ground truth.

    Returns dict with per-entry results and summary.
    """
    from sml.bible.query import Bible

    bible = Bible(bible_path)
    results = []

    # Validate HasProperty relations
    for concept_text, expected_props in GROUND_TRUTH.items():
        concept = bible.lookup_concept(concept_text)
        if concept is None:
            results.append({
                "concept": concept_text,
                "type": "HasProperty",
                "status": "MISSING_CONCEPT",
                "expected": expected_props,
                "found": [],
            })
            continue

        # Get all outgoing HasProperty relations (type_id=4)
        rels = bible.get_outgoing_relations(concept["id"])
        has_prop_rels = [r for r in rels if r["relation_type_id"] == 4]
        found_props = [r["target_text"] for r in has_prop_rels]

        missing = [p for p in expected_props if p not in found_props]
        if missing:
            status = "MISSING"
        else:
            status = "PASS"

        results.append({
            "concept": concept_text,
            "type": "HasProperty",
            "status": status,
            "expected": expected_props,
            "found": found_props,
            "missing": missing if missing else [],
        })

    # Validate CapableOf relations
    for concept_text, expected_caps in CAPABILITY_TRUTH.items():
        concept = bible.lookup_concept(concept_text)
        if concept is None:
            results.append({
                "concept": concept_text,
                "type": "CapableOf",
                "status": "MISSING_CONCEPT",
                "expected": expected_caps,
                "found": [],
            })
            continue

        rels = bible.get_outgoing_relations(concept["id"])
        cap_rels = [r for r in rels if r["relation_type_id"] == 5]
        found_caps = [r["target_text"] for r in cap_rels]

        missing = [c for c in expected_caps if c not in found_caps]
        if missing:
            status = "MISSING"
        else:
            status = "PASS"

        results.append({
            "concept": concept_text,
            "type": "CapableOf",
            "status": status,
            "expected": expected_caps,
            "found": found_caps,
            "missing": missing if missing else [],
        })

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
