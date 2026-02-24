#!/usr/bin/env python3
"""Phase 1: Critical Baselines — Run all evaluations and compare results.

Orchestrates the three key comparisons needed for Phase 1:
  1. SML fine-tuned model on SML questions (existing baseline — 86%)
  2. Vanilla model on SML questions (existing baseline — 47%)
  3. Vanilla model on NL questions (THE critical test)

Plus token efficiency measurement.

If the vanilla model scores 80%+ on NL questions, SML is an unnecessary
abstraction. If it scores significantly lower than the SML fine-tuned model,
SML provides genuine structural reasoning advantage.

Usage:
    # Full Phase 1 evaluation (all 3 comparisons)
    python sml_opaque_eval/run_phase1_eval.py \\
        --model sweetpapa/sml-qwen3-4b-phase3-full \\
        --baseline Qwen/Qwen3-4B

    # Just the NL baseline test (fastest to answer the key question)
    python sml_opaque_eval/run_phase1_eval.py \\
        --baseline Qwen/Qwen3-4B \\
        --nl-only

    # Quick test (5 examples per task)
    python sml_opaque_eval/run_phase1_eval.py \\
        --model sweetpapa/sml-qwen3-4b-phase3-full \\
        --baseline Qwen/Qwen3-4B \\
        --limit 5

    # Token efficiency only (no GPU needed)
    python sml_opaque_eval/run_phase1_eval.py --token-efficiency-only
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

SML_TASK = "sml_opaque_reasoning"
NL_TASK = "sml_nl_baseline"
SML_JSONL = SCRIPT_DIR / "sml_opaque_reasoning.jsonl"
NL_JSONL = SCRIPT_DIR / "sml_nl_baseline.jsonl"


# ── Validation ───────────────────────────────────────────────────────────────


def check_prerequisites():
    """Verify that required files exist."""
    errors = []
    if not SML_JSONL.exists():
        errors.append(
            f"  {SML_JSONL} not found — run: python sml_opaque_eval/generate_questions.py"
        )
    if not NL_JSONL.exists():
        errors.append(
            f"  {NL_JSONL} not found — run: python sml_opaque_eval/generate_nl_baseline.py"
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
    """Run lm_eval CLI for a single model/task combination.

    Returns parsed results dict or None on failure.
    """
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

    # Try to parse results from output directory
    if output_dir:
        return load_results(output_dir, task)
    return None


def load_results(output_dir: str, task: str) -> dict | None:
    """Attempt to load lm-eval results JSON from output directory."""
    out_path = Path(output_dir)
    # lm-eval saves results in subdirectories named by model
    for json_file in out_path.rglob("results.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            results = data.get("results", {})
            if task in results:
                return results[task]
        except (json.JSONDecodeError, KeyError):
            continue
    return None


# ── Token efficiency ─────────────────────────────────────────────────────────


def run_token_efficiency(tokenizer_name: str | None, output_dir: str) -> dict:
    """Run token efficiency measurement."""
    print(f"\n{'='*70}")
    print("TOKEN EFFICIENCY MEASUREMENT")
    print(f"{'='*70}")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "token_efficiency.py"),
        "--output", os.path.join(output_dir, "token_efficiency.json"),
    ]
    if tokenizer_name:
        cmd.extend(["--tokenizer", tokenizer_name])

    subprocess.run(cmd)

    # Load results
    eff_path = Path(output_dir) / "token_efficiency.json"
    if eff_path.exists():
        with open(eff_path) as f:
            return json.load(f)
    return {}


# ── Results display ──────────────────────────────────────────────────────────


def print_phase1_header(args):
    """Print the Phase 1 evaluation header."""
    print(f"\n{'#'*70}")
    print("#  PHASE 1: CRITICAL BASELINES")
    print("#  SML Opaque Reasoning — Natural Language Comparison")
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


def print_results_table(results: dict[str, dict | None]):
    """Print a comparison table of all evaluation results."""
    print(f"\n{'='*70}")
    print("PHASE 1 RESULTS SUMMARY")
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
    """Print interpretation of the Phase 1 results."""
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    nl_vanilla = results.get("Vanilla on NL questions")
    sml_tuned = results.get("SML fine-tuned on SML questions")

    if nl_vanilla and isinstance(nl_vanilla.get("acc,none", nl_vanilla.get("acc")), (int, float)):
        nl_acc = nl_vanilla.get("acc,none", nl_vanilla.get("acc", 0))

        if nl_acc >= 0.80:
            print("\n  CRITICAL: Vanilla NL baseline >= 80%!")
            print("  The vanilla model can reason from natural language context")
            print("  nearly as well as the SML-trained model from SML context.")
            print("  SML may be an unnecessary abstraction for this task.")
        elif nl_acc >= 0.60:
            print("\n  MODERATE: Vanilla NL baseline is 60-80%.")
            print("  The vanilla model has significant reasoning ability from NL.")
            print("  SML advantage is reduced but may still be meaningful.")
        elif nl_acc >= 0.40:
            print("\n  ENCOURAGING: Vanilla NL baseline is 40-60%.")
            print("  The vanilla model gets some value from NL context but")
            print("  SML-trained model likely has a significant advantage.")
        else:
            print("\n  STRONG: Vanilla NL baseline < 40%.")
            print("  NL context alone is insufficient for reasoning.")
            print("  SML provides genuine structural reasoning advantage.")

        if sml_tuned:
            sml_acc = sml_tuned.get("acc,none", sml_tuned.get("acc", 0))
            if isinstance(sml_acc, (int, float)):
                delta = sml_acc - nl_acc
                print(f"\n  SML fine-tuned ({sml_acc*100:.1f}%) vs NL vanilla ({nl_acc*100:.1f}%)")
                print(f"  Delta: {delta*100:+.1f} percentage points")
    else:
        print("\n  NL baseline results not available. Run with --baseline to test.")

    print(f"\n{'='*70}")
    print("KEY QUESTION: Does the vanilla model score 80%+ on NL questions?")
    print("  If YES → SML is an unnecessary abstraction")
    print("  If NO  → SML provides genuine structural reasoning advantage")
    print(f"{'='*70}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Critical Baselines — SML vs Natural Language comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full Phase 1 evaluation
  %(prog)s --model sweetpapa/sml-qwen3-4b-phase3-full --baseline Qwen/Qwen3-4B

  # Just the NL baseline (fastest critical test)
  %(prog)s --baseline Qwen/Qwen3-4B --nl-only

  # Quick dry run
  %(prog)s --model sweetpapa/sml-qwen3-4b-phase3-full --baseline Qwen/Qwen3-4B --limit 5

  # Token efficiency only (no GPU)
  %(prog)s --token-efficiency-only
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
        "--token-efficiency-only",
        action="store_true",
        help="Only run token efficiency measurement (no GPU needed)",
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
        default=str(SCRIPT_DIR / "results" / "phase1"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--extra-model-args",
        default="",
        help="Additional comma-separated model_args for lm-eval",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.token_efficiency_only and not args.model and not args.baseline:
        parser.error("Must specify --model, --baseline, or --token-efficiency-only")
    if args.nl_only and not args.baseline:
        parser.error("--nl-only requires --baseline")

    check_prerequisites()
    print_phase1_header(args)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    results: dict[str, dict | None] = {}

    # ── Token efficiency (always run unless skipped) ──────────────────────

    if not args.nl_only or args.token_efficiency_only:
        tokenizer_for_eff = args.baseline or args.model
        run_token_efficiency(tokenizer_for_eff, out_dir)

    if args.token_efficiency_only:
        print("\nToken efficiency measurement complete.")
        return

    # ── Evaluation runs ──────────────────────────────────────────────────

    if args.model and not args.nl_only:
        # 1. SML fine-tuned on SML questions
        r = run_lm_eval(
            model_path=args.model,
            task=SML_TASK,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            limit=args.limit,
            output_dir=os.path.join(out_dir, "sml_finetuned_sml"),
            model_type=args.model_type,
            extra_model_args=args.extra_model_args,
            label="SML fine-tuned on SML questions",
        )
        results["SML fine-tuned on SML questions"] = r

    if args.baseline and not args.nl_only:
        # 2. Vanilla on SML questions
        r = run_lm_eval(
            model_path=args.baseline,
            task=SML_TASK,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            limit=args.limit,
            output_dir=os.path.join(out_dir, "vanilla_sml"),
            model_type=args.model_type,
            extra_model_args=args.extra_model_args,
            label="Vanilla on SML questions",
        )
        results["Vanilla on SML questions"] = r

    if args.baseline:
        # 3. THE CRITICAL TEST: Vanilla on NL questions
        r = run_lm_eval(
            model_path=args.baseline,
            task=NL_TASK,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            limit=args.limit,
            output_dir=os.path.join(out_dir, "vanilla_nl"),
            model_type=args.model_type,
            extra_model_args=args.extra_model_args,
            label="Vanilla on NL questions",
        )
        results["Vanilla on NL questions"] = r

    if args.model:
        # 4. BONUS: SML fine-tuned on NL questions (how does it transfer?)
        r = run_lm_eval(
            model_path=args.model,
            task=NL_TASK,
            device=args.device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            limit=args.limit,
            output_dir=os.path.join(out_dir, "sml_finetuned_nl"),
            model_type=args.model_type,
            extra_model_args=args.extra_model_args,
            label="SML fine-tuned on NL questions",
        )
        results["SML fine-tuned on NL questions"] = r

    # ── Display results ──────────────────────────────────────────────────

    print_results_table(results)
    print_interpretation(results)

    # Save combined results
    combined_path = Path(out_dir) / "phase1_summary.json"
    with open(combined_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "results": {
                k: v for k, v in results.items() if v is not None
            },
        }, f, indent=2)
    print(f"\nPhase 1 results saved to: {combined_path}")
    print("Done!")


if __name__ == "__main__":
    main()
