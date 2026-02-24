#!/usr/bin/env python3
"""Run the SML Opaque Reasoning evaluation against HuggingFace models.

Wraps lm-evaluation-harness to run the custom sml_opaque_reasoning task,
then prints results broken down by category.

Usage:
    # Evaluate a single model
    python sml_opaque_eval/run_sml_eval.py --model sweetpapa/sml-qwen3-4b

    # Compare fine-tuned vs baseline
    python sml_opaque_eval/run_sml_eval.py \\
        --model sweetpapa/sml-qwen3-4b \\
        --baseline Qwen/Qwen3-4B

    # Custom options
    python sml_opaque_eval/run_sml_eval.py \\
        --model ./local-checkpoint \\
        --device cuda:0 \\
        --batch-size 8 \\
        --dtype float16 \\
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TASK_NAME = "sml_opaque_reasoning"
JSONL_PATH = SCRIPT_DIR / "sml_opaque_reasoning.jsonl"


# ── Category-level analysis ──────────────────────────────────────────────────


def load_questions() -> list[dict]:
    """Load the evaluation JSONL to get category metadata."""
    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} not found. Run generate_questions.py first.")
        sys.exit(1)
    with open(JSONL_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_category_header():
    """Print table header for category breakdown."""
    print(f"\n{'Category':<22} {'Count':>6} {'Random':>8}")
    print("-" * 40)


def print_category_breakdown(questions: list[dict]):
    """Print distribution of questions by category."""
    cats: dict[str, int] = {}
    for q in questions:
        cat = q.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1

    print_category_header()
    for cat, count in sorted(cats.items()):
        random_pct = 25.0  # 4-choice random
        print(f"  {cat:<20} {count:>4}     ~{random_pct:.0f}%")
    print(f"  {'TOTAL':<20} {len(questions):>4}     ~25%")


# ── lm-eval runner ───────────────────────────────────────────────────────────


def run_lm_eval(
    model_path: str,
    device: str = "cuda:0",
    batch_size: str = "auto",
    dtype: str = "float16",
    limit: int | None = None,
    output_dir: str | None = None,
    model_type: str = "hf",
    extra_model_args: str = "",
) -> Path | None:
    """Run lm_eval CLI and return the output directory path."""
    model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code=True"
    if extra_model_args:
        model_args += f",{extra_model_args}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", TASK_NAME,
        "--include_path", str(SCRIPT_DIR),
        "--device", device,
        "--batch_size", str(batch_size),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if output_dir:
        cmd.extend(["--output_path", output_dir])
        out_path = Path(output_dir)
    else:
        out_path = None

    cmd_str = " ".join(cmd)
    print(f"\n{'='*70}")
    print(f"Running: {cmd_str}")
    print(f"{'='*70}\n")

    # Run from the eval directory so relative JSONL path resolves.
    # Merge stderr into stdout so errors are always visible.
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"\nERROR: lm_eval exited with code {result.returncode}")
        return None

    return out_path


# ── Results display ──────────────────────────────────────────────────────────


def print_comparison(model_name: str, baseline_name: str | None = None):
    """Print a summary comparison header."""
    print(f"\n{'='*70}")
    print("SML OPAQUE REASONING EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"\nModel:    {model_name}")
    if baseline_name:
        print(f"Baseline: {baseline_name}")
    print(f"\nTask: {TASK_NAME} (100 multiple-choice, 4 options)")
    print(f"Random baseline: ~25%")
    print()


def print_interpretation():
    """Print how to interpret the results."""
    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("""
  ~25%     = Random chance (no SML understanding)
  25-35%   = Marginal, possibly pattern-matching
  35-50%   = Some SML reasoning ability
  50-70%   = Significant SML reasoning (approach works!)
  >70%     = Strong SML reasoning augmentation
    """)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run SML Opaque Reasoning evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model sweetpapa/sml-qwen3-4b
  %(prog)s --model ./checkpoint --baseline Qwen/Qwen3-4B
  %(prog)s --model sweetpapa/sml-qwen3-4b --device mps --limit 10
        """,
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model path or ID for the fine-tuned model",
    )
    parser.add_argument(
        "--baseline",
        help="Optional baseline model for comparison (e.g., Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--model-type", default="hf",
        help="lm-eval model type (default: hf). Use 'vllm' for vLLM backend.",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device to run on (default: cuda:0, use 'mps' for Apple Silicon)",
    )
    parser.add_argument(
        "--batch-size", default="auto",
        help="Batch size for evaluation (default: auto)",
    )
    parser.add_argument(
        "--dtype", default="float16",
        help="Model dtype (default: float16, use bfloat16 for newer GPUs)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of examples (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save lm-eval output JSON",
    )
    parser.add_argument(
        "--extra-model-args", default="",
        help="Additional comma-separated model_args for lm-eval",
    )
    args = parser.parse_args()

    # Check JSONL exists
    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} not found.")
        print("Run generate_questions.py first:")
        print(f"  python {SCRIPT_DIR / 'generate_questions.py'}")
        sys.exit(1)

    questions = load_questions()
    print_comparison(args.model, args.baseline)
    print_category_breakdown(questions)

    # Create output dir
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = str(SCRIPT_DIR / "results")
    os.makedirs(out_dir, exist_ok=True)

    # Run fine-tuned model
    print(f"\n--- Evaluating: {args.model} ---")
    run_lm_eval(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
        limit=args.limit,
        output_dir=out_dir,
        model_type=args.model_type,
        extra_model_args=args.extra_model_args,
    )

    # Run baseline if specified
    if args.baseline:
        print(f"\n--- Evaluating baseline: {args.baseline} ---")
        run_lm_eval(
            model_path=args.baseline,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            limit=args.limit,
            output_dir=out_dir,
            model_type=args.model_type,
            extra_model_args=args.extra_model_args,
        )

    print_interpretation()
    print(f"\nResults saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
