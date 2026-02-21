#!/usr/bin/env python3
"""Run lm-eval benchmarks with SML context injection.

Thin wrapper that registers the sml_hf model type, then forwards all
arguments straight to lm_eval's CLI.  This sidesteps --include_path
import issues across lm-eval versions.

SML-augmented (with Bible context injection):
    python scripts/06_benchmark_sml.py --model sml_hf --model_args pretrained=sweetpapa/sml-qwen2.5-3b-phase2,bible_path=data/sml_bible.db,dtype=float16 --tasks arc_challenge,hellaswag,truthfulqa_mc2,winogrande --device cuda:0 --batch_size 8 --num_fewshot 5 --output_path ./results/sml_qwen3_4b_sml

Baseline (no SML, same model, use plain lm_eval):
    python -m lm_eval --model hf --model_args pretrained=sweetpapa/sml-qwen2.5-3b-phase2,dtype=float16 --tasks arc_challenge,hellaswag,truthfulqa_mc2,winogrande --device cuda:0 --batch_size 8 --num_fewshot 5 --output_path ./results/sml_qwen3_4b
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so `sml` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# This import triggers @register_model("sml_hf")
import sml.evaluation.sml_harness  # noqa: F401

from lm_eval.__main__ import cli_evaluate

cli_evaluate()
