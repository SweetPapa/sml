#!/usr/bin/env python3
"""Hard Mode Evaluation: Run all hard SML reasoning evaluations and compare.

Orchestrates 4 evaluation configurations:
  1. SML fine-tuned model on hard SML questions
  2. Vanilla model on hard SML questions
  3. Vanilla model on hard NL questions
  4. SML fine-tuned model on hard NL questions

Expected target difficulty:
  - SML fine-tuned: 50-65% (vs 85% on easy)
  - Vanilla on SML: 25-35% (vs 45% on easy)
  - Vanilla on NL:  30-50% (vs 92% on easy)
  - Random: 25%

Usage:
    # Full hard evaluation
    python sml_opaque_eval/run_hard_eval.py \\
        --model sweetpapa/sml-qwen3-4b-phase3-full \\
        --baseline Qwen/Qwen3-4B

    # Quick test (5 examples per task)
    python sml_opaque_eval/run_hard_eval.py \\
        --model sweetpapa/sml-qwen3-4b-phase3-full \\
        --baseline Qwen/Qwen3-4B \\
        --limit 5

    # NL-only test
    python sml_opaque_eval/run_hard_eval.py \\
        --baseline Qwen/Qwen3-4B \\
        --nl-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

HARD_SML_TASK = "sml_hard_reasoning"
HARD_NL_TASK = "sml_hard_nl_baseline"
HARD_SML_JSONL = SCRIPT_DIR / "sml_hard_reasoning.jsonl"
HARD_NL_JSONL = SCRIPT_DIR / "sml_hard_nl_baseline.jsonl"


# ── Validation ───────────────────────────────────────────────────────────────


def check_prerequisites():
    """Verify that required files exist."""
    errors = []
    if not HARD_SML_JSONL.exists():
        errors.append(
            f"  {HARD_SML_JSONL} not found — run: "
            f"python sml_opaque_eval/generate_hard_questions.py --no-groq"
        )
    if not HARD_NL_JSONL.exists():
        errors.append(
            f"  {HARD_NL_JSONL} not found — run: "
            f"python sml_opaque_eval/generate_nl_baseline.py "
            f"--input sml_opaque_eval/sml_hard_reasoning.jsonl "
            f"--output sml_opaque_eval/sml_hard_nl_baseline.jsonl"
        )
    if errors:
        print("ERROR: Missing prerequisite files:")
        for e in errors:
            print(e)
        sys.exit(1)


# ── lm-eval runner ───────────────────────────────────────────────────────────


def run_lm_eval(
    model_path: str,
    task: str,
    device: str = "cuda:0",
    batch_size: str = "auto",
    dtype: str = "float16",
    limit: int | None = None,
    output_dir: str | None = None,
    model_type: str = "hf",
    extra_model_args: str = "",
    label: str = "",
) -> dict | None:
    """Run lm_eval CLI for a single model/task combination."""
    model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code=True"
    if extra_model_args:
        model_args += f",{extra_model_args}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", task,
        "--include_path", str(SCRIPT_DIR),
        "--device", device,
        "--batch_size", str(batch_size),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if output_dir:
        cmd.extend(["--output_path", output_dir])

    tag = label or f"{model_path} on {task}"
    print(f"\n{'='*70}")
    print(f"Running: {tag}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"\nERROR: lm_eval exited with code {result.returncode} for: {tag}")
        return None

    if output_dir:
        return load_results(output_dir, task)
    return None


def load_results(output_dir: str, task: str) -> dict | None:
    """Attempt to load lm-eval results JSON from output directory."""
    out_path = Path(output_dir)
    result_files = sorted(
        out_path.rglob("results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for json_file in result_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            results = data.get("results", {})
            if task in results:
                return results[task]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return None


# ── Results display ──────────────────────────────────────────────────────────


def print_header(args):
    """Print the evaluation header."""
    print(f"\n{'#'*70}")
    print("#  HARD MODE: SML Opaque Reasoning Evaluation")
    print("#  200 complex questions — graph traversal, inference, conflict")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    if args.model:
        print(f"\n  Fine-tuned model: {args.model}")
    if args.baseline:
        print(f"  Baseline model:   {args.baseline}")
    print(f"  Device:           {args.device}")
    print(f"  Dtype:            {args.dtype}")
    if args.limit:
        print(f"  Limit:            {args.limit} examples per task")


def print_category_breakdown():
    """Print the question category distribution."""
    if not HARD_SML_JSONL.exists():
        return

    with open(HARD_SML_JSONL) as f:
        questions = [json.loads(line) for line in f if line.strip()]

    cats: dict[str, int] = {}
    for q in questions:
        cat = q.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1

    print(f"\n{'Category':<30} {'Count':>6}")
    print("-" * 40)
    for cat in sorted(cats):
        print(f"  {cat:<28} {cats[cat]:>4}")
    print(f"  {'TOTAL':<28} {len(questions):>4}")


def print_results_table(results: dict[str, dict | None]):
    """Print a comparison table of all evaluation results."""
    print(f"\n{'='*70}")
    print("HARD MODE RESULTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Configuration':<45} {'Accuracy':>10} {'Acc Norm':>10}")
    print("-" * 68)

    for label, r in results.items():
        if r is None:
            print(f"  {label:<43} {'FAILED':>10} {'':>10}")
        else:
            acc = r.get("acc,none", r.get("acc", "N/A"))
            acc_norm = r.get("acc_norm,none", r.get("acc_norm", "N/A"))
            if isinstance(acc, (int, float)):
                acc = f"{acc*100:.1f}%"
            if isinstance(acc_norm, (int, float)):
                acc_norm = f"{acc_norm*100:.1f}%"
            print(f"  {label:<43} {acc:>10} {acc_norm:>10}")

    print(f"  {'Random baseline (4 choices)':<43} {'25.0%':>10}")


def print_interpretation(results: dict[str, dict | None]):
    """Print interpretation of the hard evaluation results."""
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    sml_tuned = results.get("SML fine-tuned on hard SML")
    vanilla_sml = results.get("Vanilla on hard SML")
    vanilla_nl = results.get("Vanilla on hard NL")
    sml_tuned_nl = results.get("SML fine-tuned on hard NL")

    def get_acc(r):
        if r is None:
            return None
        v = r.get("acc,none", r.get("acc"))
        return v if isinstance(v, (int, float)) else None

    sml_acc = get_acc(sml_tuned)
    van_sml_acc = get_acc(vanilla_sml)
    van_nl_acc = get_acc(vanilla_nl)
    sml_nl_acc = get_acc(sml_tuned_nl)

    print("\n  TARGET RANGES:")
    print("    SML fine-tuned on hard SML: 50-65%")
    print("    Vanilla on hard SML:        25-35%")
    print("    Vanilla on hard NL:         30-50%")
    print("    Random:                     25%")

    if van_nl_acc is not None:
        print(f"\n  ACTUAL vanilla NL: {van_nl_acc*100:.1f}%")
        if van_nl_acc > 0.70:
            print("  WARNING: Vanilla NL > 70% — eval may still be too easy!")
            print("  Consider: increase chain depth, add more distractors,")
            print("  or require more multi-step reasoning.")
        elif van_nl_acc > 0.50:
            print("  Vanilla NL is 50-70% — moderately challenging.")
        elif van_nl_acc > 0.30:
            print("  GOOD: Vanilla NL is 30-50% — in target range.")
        else:
            print("  Vanilla NL < 30% — near random. Eval may be too hard")
            print("  or the model cannot parse the NL context at all.")

    if sml_acc is not None:
        print(f"\n  ACTUAL SML fine-tuned: {sml_acc*100:.1f}%")
        if sml_acc < 0.35:
            print("  WARNING: SML fine-tuned < 35% — eval may be too hard!")
            print("  Consider: reduce chain depth or graph density.")
        elif sml_acc > 0.65:
            print("  SML fine-tuned > 65% — model handles hard questions well.")
        else:
            print("  GOOD: SML fine-tuned is 35-65% — in target range.")

    if sml_acc is not None and van_nl_acc is not None:
        delta = sml_acc - van_nl_acc
        print(f"\n  SML advantage: {delta*100:+.1f} percentage points")
        if delta > 0.15:
            print("  Strong SML reasoning advantage on hard questions!")
        elif delta > 0.05:
            print("  Moderate SML advantage.")
        else:
            print("  Minimal SML advantage — eval design may need adjustment.")

    print(f"\n{'='*70}")
    print("KEY QUESTION: Does hard mode differentiate SML from NL?")
    print("  If SML fine-tuned >> Vanilla NL → SML provides real advantage")
    print("  If both near random → eval too hard, reduce complexity")
    print("  If Vanilla NL > 70% → eval too easy, increase complexity")
    print(f"{'='*70}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Hard Mode: Complex SML reasoning evaluation (200 questions)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full hard evaluation
  %(prog)s --model sweetpapa/sml-qwen3-4b-phase3-full --baseline Qwen/Qwen3-4B

  # Quick test
  %(prog)s --model sweetpapa/sml-qwen3-4b-phase3-full --baseline Qwen/Qwen3-4B --limit 5

  # NL baseline only
  %(prog)s --baseline Qwen/Qwen3-4B --nl-only
        """,
    )

    parser.add_argument(
        "--model",
        help="Path or ID for the SML fine-tuned model",
    )
    parser.add_argument(
        "--baseline",
        help="Path or ID for the vanilla baseline model (e.g., Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--nl-only",
        action="store_true",
        help="Only run the NL baseline test (requires --baseline)",
    )
    parser.add_argument(
        "--model-type",
        default="hf",
        help="lm-eval model type (default: hf)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device (default: cuda:0, use 'mps' for Apple Silicon)",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit examples per task (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "hard"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--extra-model-args",
        default="",
        help="Additional comma-separated model_args for lm-eval",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.model and not args.baseline:
        parser.error("Must specify --model, --baseline, or both")
    if args.nl_only and not args.baseline:
        parser.error("--nl-only requires --baseline")

    check_prerequisites()
    print_header(args)
    print_category_breakdown()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    results: dict[str, dict | None] = {}

    # ── Evaluation runs ──────────────────────────────────────────────────

    if args.model and not args.nl_only:
        # 1. SML fine-tuned on hard SML questions
        r = run_lm_eval(
            model_path=args.model,
            task=HARD_SML_TASK,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            limit=args.limit,
            output_dir=os.path.join(out_dir, "sml_finetuned_hard_sml"),
            model_type=args.model_type,
            extra_model_args=args.extra_model_args,
            label="SML fine-tuned on hard SML",
        )
        results["SML fine-tuned on hard SML"] = r

    if args.baseline and not args.nl_only:
        # 2. Vanilla on hard SML questions
        r = run_lm_eval(
            model_path=args.baseline,
            task=HARD_SML_TASK,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            limit=args.limit,
            output_dir=os.path.join(out_dir, "vanilla_hard_sml"),
            model_type=args.model_type,
            extra_model_args=args.extra_model_args,
            label="Vanilla on hard SML",
        )
        results["Vanilla on hard SML"] = r

    if args.baseline:
        # 3. Vanilla on hard NL questions
        r = run_lm_eval(
            model_path=args.baseline,
            task=HARD_NL_TASK,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            limit=args.limit,
            output_dir=os.path.join(out_dir, "vanilla_hard_nl"),
            model_type=args.model_type,
            extra_model_args=args.extra_model_args,
            label="Vanilla on hard NL",
        )
        results["Vanilla on hard NL"] = r

    if args.model:
        # 4. SML fine-tuned on hard NL questions
        r = run_lm_eval(
            model_path=args.model,
            task=HARD_NL_TASK,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            limit=args.limit,
            output_dir=os.path.join(out_dir, "sml_finetuned_hard_nl"),
            model_type=args.model_type,
            extra_model_args=args.extra_model_args,
            label="SML fine-tuned on hard NL",
        )
        results["SML fine-tuned on hard NL"] = r

    # ── Display results ──────────────────────────────────────────────────

    print_results_table(results)
    print_interpretation(results)

    # Save combined results
    combined_path = Path(out_dir) / "hard_eval_summary.json"
    with open(combined_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "results": {
                k: v for k, v in results.items() if v is not None
            },
        }, f, indent=2)
    print(f"\nHard eval results saved to: {combined_path}")
    print("Done!")


if __name__ == "__main__":
    main()
