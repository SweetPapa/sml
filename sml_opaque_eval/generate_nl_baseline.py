#!/usr/bin/env python3
"""Generate Natural Language baseline dataset from the SML opaque reasoning questions.

Takes the existing sml_opaque_reasoning.jsonl (100 SML questions) and converts
each <sml> block into an equivalent natural language description.  This is the
critical Phase 1 baseline: if vanilla Qwen3-4B scores 80%+ on the NL version
without fine-tuning, SML is an unnecessary abstraction.

Usage:
    python sml_opaque_eval/generate_nl_baseline.py
    python sml_opaque_eval/generate_nl_baseline.py --input custom.jsonl
    python sml_opaque_eval/generate_nl_baseline.py --output custom_nl.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Relation type → natural language mapping ─────────────────────────────────

RELATION_NL = {
    "IsA": "is a type of",
    "HasA": "has",
    "PartOf": "is part of",
    "AtLocation": "is located at",
    "CapableOf": "is capable of",
    "UsedFor": "is used for",
    "HasProperty": "has the property",
    "Causes": "causes",
    "HasPrerequisite": "has the prerequisite",
    "RelatedTo": "is related to",
    "SimilarTo": "is similar to",
    "Antonym": "is the opposite of",
    "Synonym": "is a synonym of",
    "MadeOf": "is made of",
    "DerivedFrom": "is derived from",
    "MannerOf": "is a manner of",
    "MotivatedByGoal": "is motivated by",
    "FormOf": "is a form of",
    # Negated versions
    "NOT_IsA": "is NOT a type of",
    "NOT_HasA": "does NOT have",
    "NOT_PartOf": "is NOT part of",
    "NOT_AtLocation": "is NOT located at",
    "NOT_CapableOf": "is NOT capable of",
    "NOT_UsedFor": "is NOT used for",
    "NOT_HasProperty": "does NOT have the property",
    "NOT_Causes": "does NOT cause",
    "NOT_HasPrerequisite": "does NOT have the prerequisite",
    "NOT_RelatedTo": "is NOT related to",
    "NOT_SimilarTo": "is NOT similar to",
}

# ── SML parsing ──────────────────────────────────────────────────────────────

# Match E(domain|category|subcategory|specificity|anchor|modifier|temporal|confidence)
ENTITY_RE = re.compile(
    r"E\((\d+)\|(\d+)\|(\d+)\|(\d+)\|([^|]+)\|([^|]+)\|(\d+)\|([0-9.]+)\)"
)

# Match R(RelType|src|tgt|weight|temporal|negation)
RELATION_RE = re.compile(
    r"R\(([A-Za-z_]+)\|(\d+)\|(\d+)\|([0-9.]+)\|(\d+)\|(\d+)\)"
)


def parse_sml_block(sml_text: str) -> tuple[list[dict], list[dict]]:
    """Parse an SML block into entity and relation dicts.

    Returns:
        (entities, relations) where each is a list of dicts with parsed fields.
    """
    entities = []
    for m in ENTITY_RE.finditer(sml_text):
        entities.append({
            "index": len(entities),
            "anchor": m.group(5),
            "confidence": float(m.group(8)),
        })

    relations = []
    for m in RELATION_RE.finditer(sml_text):
        relations.append({
            "rel_type": m.group(1),
            "source": int(m.group(2)),
            "target": int(m.group(3)),
            "weight": float(m.group(4)),
        })

    return entities, relations


def sml_to_natural_language(sml_text: str) -> str:
    """Convert an SML block to equivalent natural language.

    Example output:
        The following entities and relationships are defined:
        Entities: X0, X1, X2
        Relationships:
        - X0 is a type of X1 (confidence: 0.85)
        - X1 is capable of X2 (confidence: 0.90)
    """
    entities, relations = parse_sml_block(sml_text)

    if not entities:
        return sml_text  # fallback: return unchanged

    lines = []
    lines.append("The following entities and relationships are defined:")

    # Entity list
    entity_names = [e["anchor"] for e in entities]
    lines.append(f"Entities: {', '.join(entity_names)}")

    if relations:
        lines.append("Relationships:")
        for rel in relations:
            rel_type = rel["rel_type"]
            src_name = entities[rel["source"]]["anchor"] if rel["source"] < len(entities) else f"E{rel['source']}"
            tgt_name = entities[rel["target"]]["anchor"] if rel["target"] < len(entities) else f"E{rel['target']}"
            weight = rel["weight"]

            nl_phrase = RELATION_NL.get(rel_type, rel_type.replace("_", " ").lower())
            lines.append(f"- {src_name} {nl_phrase} {tgt_name} (confidence: {weight})")
    else:
        lines.append("No relationships are defined.")

    return "\n".join(lines)


def clean_sml_references(text: str) -> str:
    """Replace SML-specific language in question text with format-neutral phrasing.

    This ensures the NL baseline questions don't reference a format the model
    isn't seeing, which could confuse it or provide an unfair disadvantage.
    """
    replacements = [
        ("According to the SML data, ", "According to the data above, "),
        ("According to the SML data ", "According to the data above "),
        ("Based on the SML data, ", "Based on the data above, "),
        ("Based on the SML data ", "Based on the data above "),
        ("Based on the SML relations, ", "Based on the relationships above, "),
        ("Based on the SML relations ", "Based on the relationships above "),
        ("in the SML block", "in the data above"),
        ("the SML block", "the data above"),
        ("The SML block", "The data above"),
        ("the SML data", "the data above"),
        ("The SML data", "The data above"),
        ("No negation is specified in the SML data", "No negation is specified in the data above"),
        ("Not specified in the SML data", "Not specified in the data above"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def convert_question(question: dict) -> dict:
    """Convert a single SML question to its NL equivalent.

    Replaces the <sml>...</sml> block in the question text with
    natural language, and cleans up SML-specific references in the
    question phrasing.
    """
    full_text = question["question"]

    # Extract the SML block
    sml_match = re.search(r"<sml>\n(.*?)\n</sml>", full_text, re.DOTALL)
    if not sml_match:
        # No SML block found — return as-is
        return {**question, "format": "nl_baseline"}

    sml_block = sml_match.group(0)  # includes <sml> tags
    sml_content = sml_match.group(1)  # just the content

    # Convert SML block to NL
    nl_text = sml_to_natural_language(sml_content)

    # Replace SML block with NL text
    new_question_text = full_text.replace(sml_block, nl_text)

    # Clean up SML-specific references in question text
    new_question_text = clean_sml_references(new_question_text)

    # Also clean up choices that reference SML
    new_choices = [clean_sml_references(c) for c in question["choices"]]

    return {
        **question,
        "question": new_question_text,
        "choices": new_choices,
        "format": "nl_baseline",
        "original_sml": sml_block,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate Natural Language baseline from SML opaque reasoning questions"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SCRIPT_DIR / "sml_opaque_reasoning.jsonl"),
        help="Input SML JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "sml_nl_baseline.jsonl"),
        help="Output NL JSONL path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run generate_questions.py first.")
        sys.exit(1)

    # Load SML questions
    with open(input_path) as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(questions)} SML questions from {input_path}")

    # Convert each to NL
    nl_questions = []
    for q in questions:
        nl_q = convert_question(q)
        nl_questions.append(nl_q)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for q in nl_questions:
            f.write(json.dumps(q) + "\n")

    print(f"Wrote {len(nl_questions)} NL baseline questions to {output_path}")

    # Print a few examples for verification
    print("\n--- Example conversions ---")
    for i in [0, 20, 50, 80]:
        if i < len(nl_questions):
            q = nl_questions[i]
            print(f"\n[Q{i+1}] Category: {q['category']}")
            # Show just the first few lines of the question
            lines = q["question"].split("\n")
            preview = "\n".join(lines[:8])
            if len(lines) > 8:
                preview += "\n..."
            print(preview)

    # Category breakdown
    cats: dict[str, int] = {}
    for q in nl_questions:
        cat = q.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    print(f"\nCategory distribution: {json.dumps(cats, indent=2)}")
    print("Done!")


if __name__ == "__main__":
    main()
