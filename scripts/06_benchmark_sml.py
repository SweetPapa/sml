#!/usr/bin/env python3
"""Run lm-evaluation-harness benchmarks with SML context injection.

Examples:
    # SML-augmented evaluation (default tasks)
    python scripts/06_benchmark_sml.py --model sweetpapa/sml-qwen2.5-3b-phase2

    # Specific tasks
    python scripts/06_benchmark_sml.py --model sweetpapa/sml-qwen2.5-3b-phase2 \
        --tasks hellaswag,arc_easy,piqa

    # Compare SML vs. baseline (runs both, prints delta)
    python scripts/06_benchmark_sml.py --model sweetpapa/sml-qwen2.5-3b-phase2 --compare

    # Baseline only (no SML, plain HFLM)
    python scripts/06_benchmark_sml.py --model sweetpapa/sml-qwen2.5-3b-phase2 --baseline-only
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sml.config import BIBLE_DB_PATH, DATA_DIR

DEFAULT_TASKS = "hellaswag,arc_easy,piqa,boolq"


def run_sml_eval(args, tasks):
    """Run evaluation with SML context injection."""
    import lm_eval
    from sml.evaluation.sml_harness import SMLAugmentedHFLM

    print("Loading model with SML augmentation...")
    model = SMLAugmentedHFLM(
        pretrained=args.model,
        bible_path=args.bible,
        batch_size=args.batch_size,
        max_encode=args.max_encode,
    )

    print(f"Running SML-augmented eval: {tasks}")
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    cache = model.sml_cache_info
    print(f"\nSML cache: {cache.hits} hits / {cache.misses} misses "
          f"({cache.hits / max(cache.hits + cache.misses, 1):.0%} hit rate)")

    # Clean up encoder resources
    del model
    return results


def run_baseline_eval(args, tasks):
    """Run evaluation without SML (plain HFLM)."""
    import lm_eval

    print("Running baseline eval (no SML)...")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model}",
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    return results


def extract_scores(results):
    """Pull per-task accuracy from lm-eval results dict."""
    scores = {}
    for task_name, task_data in results["results"].items():
        # lm-eval stores metrics like "acc,none", "acc_norm,none", etc.
        for metric_key, value in task_data.items():
            if metric_key.startswith("acc"):
                scores[f"{task_name}/{metric_key}"] = value
    return scores


def print_results(results, label="Results"):
    """Pretty-print lm-eval results."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    for task_name, task_data in results["results"].items():
        print(f"\n  {task_name}:")
        for metric_key, value in sorted(task_data.items()):
            if isinstance(value, float):
                print(f"    {metric_key}: {value:.4f}")
            elif metric_key != "alias":
                print(f"    {metric_key}: {value}")


def print_comparison(sml_scores, baseline_scores):
    """Print side-by-side comparison."""
    print(f"\n{'=' * 70}")
    print("  SML vs. Baseline Comparison")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<40s} {'Baseline':>10s} {'SML':>10s} {'Delta':>10s}")
    print(f"  {'-' * 70}")

    all_keys = sorted(set(list(sml_scores.keys()) + list(baseline_scores.keys())))
    for key in all_keys:
        b = baseline_scores.get(key)
        s = sml_scores.get(key)
        if b is not None and s is not None:
            delta = s - b
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<40s} {b:>10.4f} {s:>10.4f} {sign}{delta:>9.4f}")
        elif s is not None:
            print(f"  {key:<40s} {'N/A':>10s} {s:>10.4f}")
        elif b is not None:
            print(f"  {key:<40s} {b:>10.4f} {'N/A':>10s}")


def main():
    parser = argparse.ArgumentParser(
        description="lm-evaluation-harness with SML context injection",
    )
    parser.add_argument(
        "--model", type=str,
        default="sweetpapa/sml-qwen2.5-3b-phase2",
        help="HuggingFace model ID or local path to the fine-tuned model",
    )
    parser.add_argument(
        "--bible", type=str, default=str(BIBLE_DB_PATH),
        help="Path to the SML Bible database",
    )
    parser.add_argument(
        "--tasks", type=str, default=DEFAULT_TASKS,
        help=f"Comma-separated lm-eval task names (default: {DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=0,
        help="Number of few-shot examples (default: 0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--limit", type=float, default=None,
        help="Limit number of examples per task (int) or fraction (float <1)",
    )
    parser.add_argument(
        "--max-encode", type=int, default=2048,
        help="Max chars of the prompt tail to send through the SML encoder",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both SML and baseline, print delta",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Run baseline only (no SML injection)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(DATA_DIR / "benchmark_results"),
        help="Directory for JSON output",
    )
    args = parser.parse_args()

    # Dependency check
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print("Error: lm-eval not installed.")
        print("  pip install lm-eval")
        sys.exit(1)

    # Path checks (skip for HuggingFace hub IDs like "user/repo")
    if "/" not in args.model and not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    if not args.baseline_only and not Path(args.bible).exists():
        print(f"Error: Bible not found at {args.bible}")
        sys.exit(1)

    tasks = [t.strip() for t in args.tasks.split(",")]
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SML Benchmark Runner")
    print("=" * 70)
    print(f"  Model:     {args.model}")
    print(f"  Tasks:     {tasks}")
    print(f"  Few-shot:  {args.num_fewshot}")
    print(f"  Limit:     {args.limit or 'full'}")
    mode = "baseline" if args.baseline_only else ("compare" if args.compare else "sml")
    print(f"  Mode:      {mode}")
    print()

    sml_results = None
    baseline_results = None

    # ── Run evaluations ────────────────────────────────────────────────

    if args.baseline_only:
        baseline_results = run_baseline_eval(args, tasks)
        print_results(baseline_results, "Baseline (no SML)")
    elif args.compare:
        # Baseline first (lighter), then SML
        baseline_results = run_baseline_eval(args, tasks)
        print_results(baseline_results, "Baseline (no SML)")

        sml_results = run_sml_eval(args, tasks)
        print_results(sml_results, "SML-Augmented")

        # Delta
        sml_scores = extract_scores(sml_results)
        baseline_scores = extract_scores(baseline_results)
        print_comparison(sml_scores, baseline_scores)
    else:
        sml_results = run_sml_eval(args, tasks)
        print_results(sml_results, "SML-Augmented")

    # ── Save results ───────────────────────────────────────────────────

    output_data = {
        "run_id": run_id,
        "model": args.model,
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "mode": mode,
    }
    if sml_results:
        output_data["sml_results"] = sml_results["results"]
    if baseline_results:
        output_data["baseline_results"] = baseline_results["results"]

    output_path = output_dir / f"bench_{run_id}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
