# SML Opaque Evaluation Suite

Custom evaluation suite for testing SML (Structured Markup Language) reasoning capabilities.  Uses completely opaque entity tokens (X0, X1, X2...) so the model has zero pretrained knowledge about the entities — the **only** source of information is the relational structure.

## Overview

100 multiple-choice questions (4 choices each, 25% random baseline) across six categories:

| Category          | Count | Tests                                                |
|:------------------|------:|:-----------------------------------------------------|
| Simple Lookup     |    20 | Reading a single relation to answer a direct question |
| Negation          |    15 | Interpreting NOT_ relations correctly                 |
| Multi-Hop         |    20 | Following 2-3 relation chains to reach conclusions    |
| Weight Comparison |    15 | Comparing relation weights to determine strength      |
| Counting          |    10 | Counting entities, relations, structural properties   |
| Composite         |    20 | Combining multiple reasoning skills                   |

## File Structure

```
sml_opaque_eval/
├── README.md                        # This file
│
│── Existing evaluation (proof of concept):
├── sml_opaque_reasoning.yaml        # lm-eval task: SML format questions
├── sml_opaque_reasoning.jsonl       # 100 SML-formatted questions
├── generate_questions.py            # Generates the SML questions
├── run_sml_eval.py                  # Runs SML eval against models
│
│── Phase 1: Critical Baselines
├── sml_nl_baseline.yaml             # lm-eval task: NL format questions
├── sml_nl_baseline.jsonl            # 100 NL-converted questions (same content)
├── generate_nl_baseline.py          # Converts SML questions → NL format
├── token_efficiency.py              # Measures SML vs NL token efficiency
└── run_phase1_eval.py               # Orchestrates all Phase 1 comparisons
```

## Quick Start

### Prerequisites

```bash
pip install lm-eval>=0.4.4 transformers torch
```

A GPU with at least 10GB VRAM is required for model evaluation. Use `--device mps` for Apple Silicon.

### Generate Evaluation Data

The SML questions (`sml_opaque_reasoning.jsonl`) are already checked in.  To regenerate or modify:

```bash
# Regenerate SML questions (uses seed 42 for reproducibility)
python sml_opaque_eval/generate_questions.py --no-groq

# Generate NL baseline from existing SML questions
python sml_opaque_eval/generate_nl_baseline.py
```

The NL baseline file (`sml_nl_baseline.jsonl`) is also checked in.

---

## Phase 1: Critical Baselines

Phase 1 answers two questions:

1. **Can a vanilla model reason from natural language descriptions as well as an SML-trained model from SML format?**  If vanilla scores 80%+ on NL questions, SML is an unnecessary abstraction.

2. **Is SML more token-efficient than equivalent natural language?**  If SML is more compact, there's a practical argument even if accuracy is similar.

### Run Full Phase 1 Evaluation

This runs all four model/task combinations and token efficiency:

```bash
python sml_opaque_eval/run_phase1_eval.py \
    --model sweetpapa/sml-qwen3-4b-phase3-full \
    --baseline Qwen/Qwen3-4B \
    --device cuda:0
```

**Configurations tested:**

| # | Model                | Task Format | Purpose                             |
|---|:---------------------|:------------|:------------------------------------|
| 1 | SML fine-tuned       | SML         | Established result (~86%)           |
| 2 | Vanilla Qwen3-4B     | SML         | Established result (~47%)           |
| 3 | Vanilla Qwen3-4B     | **NL**      | **THE critical test**               |
| 4 | SML fine-tuned       | NL          | Transfer test (does SML help on NL?)|

### Run Just the NL Baseline (Fastest Critical Test)

If you only want to answer the key question:

```bash
python sml_opaque_eval/run_phase1_eval.py \
    --baseline Qwen/Qwen3-4B \
    --nl-only \
    --device cuda:0
```

### Run Token Efficiency Only (No GPU Needed)

```bash
# With default tiktoken tokenizer
python sml_opaque_eval/token_efficiency.py

# With the actual Qwen tokenizer (more accurate)
python sml_opaque_eval/token_efficiency.py --tokenizer Qwen/Qwen3-4B

# Save results to JSON
python sml_opaque_eval/token_efficiency.py \
    --tokenizer Qwen/Qwen3-4B \
    --output results/token_efficiency.json
```

### Quick Dry Run (Verify Setup)

Test with a small subset to make sure everything works:

```bash
python sml_opaque_eval/run_phase1_eval.py \
    --model sweetpapa/sml-qwen3-4b-phase3-full \
    --baseline Qwen/Qwen3-4B \
    --limit 5 \
    --device cuda:0
```

### Apple Silicon (MPS)

```bash
python sml_opaque_eval/run_phase1_eval.py \
    --baseline Qwen/Qwen3-4B \
    --nl-only \
    --device mps \
    --dtype float32
```

Note: MPS may require `float32` instead of `float16`.

---

## Running Individual Evaluations

### SML Format Evaluation (Original)

```bash
# Direct lm-eval command
lm_eval \
    --model hf \
    --model_args pretrained=sweetpapa/sml-qwen3-4b-phase3-full,dtype=float16,trust_remote_code=True \
    --tasks sml_opaque_reasoning \
    --include_path sml_opaque_eval/ \
    --device cuda:0

# Or via the runner script
python sml_opaque_eval/run_sml_eval.py \
    --model sweetpapa/sml-qwen3-4b-phase3-full \
    --baseline Qwen/Qwen3-4B
```

### NL Format Evaluation

```bash
# Direct lm-eval command
lm_eval \
    --model hf \
    --model_args pretrained=Qwen/Qwen3-4B,dtype=float16,trust_remote_code=True \
    --tasks sml_nl_baseline \
    --include_path sml_opaque_eval/ \
    --device cuda:0
```

---

## Interpreting Results

### Phase 1 Decision Matrix

| Vanilla NL Score | Interpretation                                        | Action                        |
|:-----------------|:------------------------------------------------------|:------------------------------|
| >= 80%           | SML is unnecessary; NL context is sufficient           | Thesis needs fundamental pivot |
| 60-79%           | NL is strong but SML may still have edge               | Investigate per-category gaps  |
| 40-59%           | SML-trained model has meaningful advantage              | Thesis supported; continue     |
| 25-39%           | NL context barely helps; SML is clearly superior       | Strong result; publish         |

### Expected Baselines

- **Random chance:** 25% (4 choices)
- **SML fine-tuned on SML:** ~86% (established)
- **Vanilla on SML:** ~47% (established, some surface pattern matching)
- **Vanilla on NL:** **This is what we're testing**

### Token Efficiency

The token efficiency report compares the context overhead:

- **SML tokens:** How many tokens the `<sml>` block uses
- **NL tokens:** How many tokens the equivalent NL description uses
- **Ratio:** NL/SML — values > 1 mean SML is more compact

---

## Output

Results are saved to `sml_opaque_eval/results/phase1/`:

```
results/phase1/
├── phase1_summary.json         # Combined results + interpretation
├── token_efficiency.json       # Detailed token analysis
├── sml_finetuned_sml/          # lm-eval output for config 1
├── vanilla_sml/                # lm-eval output for config 2
├── vanilla_nl/                 # lm-eval output for config 3
└── sml_finetuned_nl/           # lm-eval output for config 4
```

---

## Technical Details

### NL Conversion

The `generate_nl_baseline.py` script converts SML blocks like:

```
<sml>
E(0|0|0|0|X0|0|0|0.9)
E(0|0|0|0|X1|0|0|0.9)
E(0|0|0|0|X2|0|0|0.9)
R(IsA|0|1|0.85|0|0)
R(CapableOf|1|2|0.90|0|0)
</sml>
```

Into equivalent natural language:

```
The following entities and relationships are defined:
Entities: X0, X1, X2
Relationships:
- X0 is a type of X1 (confidence: 0.85)
- X1 is capable of X2 (confidence: 0.9)
```

The conversion preserves:
- All entity references (X0, X1, etc.)
- All relation types and directions
- All confidence weights
- Question text (with "SML" references cleaned to "data above")
- Answer choices and correct answer indices

### Evaluation Method

All evaluations use **log-likelihood scoring** (not generative). The model assigns probabilities to each answer choice (A, B, C, D) and the highest-probability choice is selected. This is the standard approach used by benchmarks like ARC and HellaSwag.

### Reproducibility

- SML questions generated with `--seed 42`
- NL questions are a deterministic transformation of SML questions
- All JSONL files are checked into the repo
