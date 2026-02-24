#!/usr/bin/env python3
"""Measure token efficiency: SML blocks vs equivalent natural language.

Compares the token count of SML-formatted context blocks against their
natural language equivalents across all 100 evaluation questions.  If SML
is more compact, this creates a practical argument for context-window-
constrained applications even if accuracy is similar.

Usage:
    python sml_opaque_eval/token_efficiency.py
    python sml_opaque_eval/token_efficiency.py --tokenizer Qwen/Qwen3-4B
    python sml_opaque_eval/token_efficiency.py --output results/token_efficiency.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def count_chars(text: str) -> int:
    """Count characters in text."""
    return len(text)


def count_tokens_tiktoken(text: str) -> int:
    """Fallback token count using tiktoken cl100k_base (GPT-4 tokenizer).

    Used when the model-specific tokenizer is not available.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Very rough estimate: ~4 chars per token
        return len(text) // 4


def count_tokens_hf(text: str, tokenizer) -> int:
    """Count tokens using a HuggingFace tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_sml_block(question_text: str) -> str | None:
    """Extract the <sml>...</sml> block from a question."""
    match = re.search(r"<sml>\n.*?\n</sml>", question_text, re.DOTALL)
    return match.group(0) if match else None


def extract_nl_context(question_text: str) -> str | None:
    """Extract the NL context block from a converted question.

    The NL context is everything before the first blank line that separates
    it from the question text.
    """
    # NL questions start with "The following entities..."
    if not question_text.startswith("The following"):
        return None
    # Find the double-newline that separates context from question
    parts = question_text.split("\n\n", 1)
    return parts[0] if parts else None


def analyze_pair(
    sml_q: dict,
    nl_q: dict,
    tokenizer=None,
) -> dict:
    """Analyze a single SML/NL question pair for token efficiency."""
    sml_block = extract_sml_block(sml_q["question"])
    nl_context = extract_nl_context(nl_q["question"])

    if not sml_block or not nl_context:
        return {}

    sml_chars = count_chars(sml_block)
    nl_chars = count_chars(nl_context)

    if tokenizer:
        sml_tokens = count_tokens_hf(sml_block, tokenizer)
        nl_tokens = count_tokens_hf(nl_context, tokenizer)
    else:
        sml_tokens = count_tokens_tiktoken(sml_block)
        nl_tokens = count_tokens_tiktoken(nl_context)

    return {
        "category": sml_q.get("category", "unknown"),
        "num_entities": sml_q.get("num_entities", 0),
        "num_relations": sml_q.get("num_relations", 0),
        "sml_chars": sml_chars,
        "nl_chars": nl_chars,
        "sml_tokens": sml_tokens,
        "nl_tokens": nl_tokens,
        "char_ratio": nl_chars / sml_chars if sml_chars > 0 else 0,
        "token_ratio": nl_tokens / sml_tokens if sml_tokens > 0 else 0,
        "token_savings": nl_tokens - sml_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure token efficiency: SML vs Natural Language"
    )
    parser.add_argument(
        "--sml-input",
        type=str,
        default=str(SCRIPT_DIR / "sml_opaque_reasoning.jsonl"),
        help="Path to SML questions JSONL",
    )
    parser.add_argument(
        "--nl-input",
        type=str,
        default=str(SCRIPT_DIR / "sml_nl_baseline.jsonl"),
        help="Path to NL baseline questions JSONL",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer to use (e.g., Qwen/Qwen3-4B). "
             "Falls back to tiktoken cl100k_base if not specified.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (optional)",
    )
    args = parser.parse_args()

    # Load datasets
    sml_path = Path(args.sml_input)
    nl_path = Path(args.nl_input)

    if not sml_path.exists():
        print(f"ERROR: {sml_path} not found.")
        sys.exit(1)
    if not nl_path.exists():
        print(f"ERROR: {nl_path} not found. Run generate_nl_baseline.py first.")
        sys.exit(1)

    with open(sml_path) as f:
        sml_qs = [json.loads(line) for line in f if line.strip()]
    with open(nl_path) as f:
        nl_qs = [json.loads(line) for line in f if line.strip()]

    assert len(sml_qs) == len(nl_qs), (
        f"Mismatch: {len(sml_qs)} SML vs {len(nl_qs)} NL questions"
    )

    # Load tokenizer if specified
    tokenizer = None
    tokenizer_name = "tiktoken/cl100k_base"
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer: {args.tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer, trust_remote_code=True
            )
            tokenizer_name = args.tokenizer
            print("Tokenizer loaded.")
        except Exception as e:
            print(f"WARNING: Could not load tokenizer {args.tokenizer}: {e}")
            print("Falling back to tiktoken cl100k_base.")

    # Analyze all pairs
    results = []
    for sml_q, nl_q in zip(sml_qs, nl_qs):
        r = analyze_pair(sml_q, nl_q, tokenizer)
        if r:
            results.append(r)

    if not results:
        print("ERROR: No valid question pairs found.")
        sys.exit(1)

    # ── Summary statistics ────────────────────────────────────────────────

    total_sml_tokens = sum(r["sml_tokens"] for r in results)
    total_nl_tokens = sum(r["nl_tokens"] for r in results)
    total_sml_chars = sum(r["sml_chars"] for r in results)
    total_nl_chars = sum(r["nl_chars"] for r in results)
    avg_token_ratio = sum(r["token_ratio"] for r in results) / len(results)
    avg_char_ratio = sum(r["char_ratio"] for r in results) / len(results)

    print(f"\n{'='*70}")
    print("TOKEN EFFICIENCY: SML vs Natural Language")
    print(f"{'='*70}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Questions analyzed: {len(results)}")

    print(f"\n{'Metric':<30} {'SML':>10} {'NL':>10} {'Ratio':>10}")
    print("-" * 65)
    print(f"{'Total tokens':<30} {total_sml_tokens:>10,} {total_nl_tokens:>10,} {total_nl_tokens/total_sml_tokens:>10.2f}x")
    print(f"{'Total characters':<30} {total_sml_chars:>10,} {total_nl_chars:>10,} {total_nl_chars/total_sml_chars:>10.2f}x")
    print(f"{'Avg tokens per question':<30} {total_sml_tokens/len(results):>10.1f} {total_nl_tokens/len(results):>10.1f} {avg_token_ratio:>10.2f}x")
    print(f"{'Avg chars per question':<30} {total_sml_chars/len(results):>10.1f} {total_nl_chars/len(results):>10.1f} {avg_char_ratio:>10.2f}x")

    if total_sml_tokens < total_nl_tokens:
        savings_pct = (1 - total_sml_tokens / total_nl_tokens) * 100
        print(f"\nSML is {savings_pct:.1f}% more token-efficient than NL")
    else:
        overhead_pct = (total_sml_tokens / total_nl_tokens - 1) * 100
        print(f"\nSML uses {overhead_pct:.1f}% MORE tokens than NL")

    # ── Per-category breakdown ────────────────────────────────────────────

    cats: dict[str, list[dict]] = {}
    for r in results:
        cat = r["category"]
        cats.setdefault(cat, []).append(r)

    print(f"\n{'Category':<22} {'SML tok':>9} {'NL tok':>9} {'Ratio':>8} {'Savings':>9}")
    print("-" * 62)
    for cat in sorted(cats):
        items = cats[cat]
        s = sum(r["sml_tokens"] for r in items)
        n = sum(r["nl_tokens"] for r in items)
        ratio = n / s if s else 0
        savings = n - s
        print(f"  {cat:<20} {s:>7,} {n:>7,} {ratio:>7.2f}x {savings:>+8,}")

    # ── By complexity (entity count) ──────────────────────────────────────

    print(f"\n{'Entities':<12} {'SML tok':>9} {'NL tok':>9} {'Ratio':>8} {'Count':>7}")
    print("-" * 50)
    ent_groups: dict[int, list[dict]] = {}
    for r in results:
        ne = r["num_entities"]
        ent_groups.setdefault(ne, []).append(r)
    for ne in sorted(ent_groups):
        items = ent_groups[ne]
        s = sum(r["sml_tokens"] for r in items) / len(items)
        n = sum(r["nl_tokens"] for r in items) / len(items)
        ratio = n / s if s else 0
        print(f"  {ne:<10} {s:>8.1f} {n:>8.1f} {ratio:>7.2f}x {len(items):>6}")

    # ── Save results ──────────────────────────────────────────────────────

    summary = {
        "tokenizer": tokenizer_name,
        "num_questions": len(results),
        "total_sml_tokens": total_sml_tokens,
        "total_nl_tokens": total_nl_tokens,
        "total_sml_chars": total_sml_chars,
        "total_nl_chars": total_nl_chars,
        "avg_token_ratio": round(avg_token_ratio, 4),
        "avg_char_ratio": round(avg_char_ratio, 4),
        "sml_more_efficient": total_sml_tokens < total_nl_tokens,
        "per_category": {
            cat: {
                "sml_tokens": sum(r["sml_tokens"] for r in items),
                "nl_tokens": sum(r["nl_tokens"] for r in items),
                "ratio": round(sum(r["nl_tokens"] for r in items) / max(1, sum(r["sml_tokens"] for r in items)), 4),
                "count": len(items),
            }
            for cat, items in sorted(cats.items())
        },
        "per_question": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nDetailed results saved to: {out_path}")

    print("\nDone!")
    return summary


if __name__ == "__main__":
    main()
