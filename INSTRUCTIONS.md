# SML Pipeline — Windows RTX 4090 Setup Guide

## Prerequisites

- Windows 10/11 with NVIDIA RTX 4090
- Python 3.10+ (recommend 3.10.x for best compatibility)
- CUDA Toolkit 12.1+ (https://developer.nvidia.com/cuda-downloads)
- Git
- Groq API key (https://console.groq.com)

## Step 1: Clone and Setup

```bash
git clone <your-repo-url> sml
cd sml
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA support FIRST (required for RTX 4090)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Then install everything else
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Why the separate torch install?** Plain `pip install torch` on Windows gets the CPU-only build. The `--index-url` flag pulls the CUDA 12.6 build, which is what the RTX 4090 needs. If you have a different CUDA version, see https://pytorch.org/get-started/locally/ for the right URL.

## Step 2: Set Environment Variables

```bash
set GROQ_API_KEY=your_groq_api_key_here
```

Or copy the example `.env` file and fill in your key:
```bash
copy .env.example .env
# Then edit .env with your key
```

## Step 3: Build the SML Bible

### Micro-PoC (~50 concepts, instant)
```bash
python scripts/01_build_bible.py --mode micro
```

### Full Build (~100K concepts, requires internet + ~30 min)
```bash
python scripts/01_build_bible.py --mode full
```

**Expected output:** `data/sml_bible.db` — SQLite database with concepts and relations.

## Step 4: Generate Training Data

```bash
python scripts/02_generate_data.py --num-examples 200 --validate
```

For more examples (recommended for better results):
```bash
python scripts/02_generate_data.py --num-examples 500 --validate
```

**Expected output:** `data/training_data.jsonl` — JSONL file with ChatML training tuples.

**Note:** Generation uses Groq API with rate limiting (~2 sec/example). 200 examples ≈ 7 minutes.

## Step 5: Fine-Tune the Model

```bash
python scripts/03_train.py --epochs 3 --merge
```

Adjust for your needs:
```bash
# Quick test (fewer epochs)
python scripts/03_train.py --epochs 1 --merge

# More training
python scripts/03_train.py --epochs 5 --batch-size 4 --merge
```

**Expected output:**
- `data/model_output/sml_adapter/` — LoRA adapter weights
- `data/model_output/sml_merged/` — Full merged model (if --merge used)

**VRAM usage:** ~12-14GB with default settings (QLoRA 4-bit).

## Step 6: Interactive Inference

```bash
python scripts/04_inference.py
```

Or single query:
```bash
python scripts/04_inference.py --query "What color is the sun?"
```

## Step 7: Run Evaluation (Liar Ablation)

```bash
python scripts/05_evaluate.py --verbose
```

**What to look for:**
- **Normal tests**: Model answers correctly using SML grounding → PASS
- **Liar tests**: Model follows the *false* SML (says "sun is green") → PASS = grounding works!
- **Unknown tests**: Model doesn't crash on unknown concepts → PASS

If Liar tests pass ≥50%, **SML grounding is working** — the model trusts the SML over its own weights.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce `--batch-size` to 2 or 1 |
| `ModuleNotFoundError: unsloth` | `pip install "unsloth[colab-new]>=2024.4"` |
| `spacy model not found` | `python -m spacy download en_core_web_sm` |
| `GROQ_API_KEY not set` | `set GROQ_API_KEY=your_key` or pass `--groq-api-key` |
| `Bible not found` | Run `python scripts/01_build_bible.py --mode micro` first |
| Groq rate limit errors | Wait 60 seconds, re-run. Or use `--num-examples` with a smaller count |
| Training loss not decreasing | Increase `--epochs` to 5 or increase training data to 500+ examples |

## Pipeline Summary

```
01_build_bible.py  → data/sml_bible.db         (concepts + relations)
02_generate_data.py → data/training_data.jsonl   (ChatML training tuples)
03_train.py        → data/model_output/          (LoRA adapter + merged model)
04_inference.py    → interactive Q&A              (SML-grounded responses)
05_evaluate.py     → evaluation report            (Liar Ablation results)
```

## Quick Start (Micro-PoC, ~15 minutes total)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m spacy download en_core_web_sm
copy .env.example .env   # then edit with your Groq API key
python scripts/01_build_bible.py --mode micro
python scripts/02_generate_data.py --num-examples 200 --validate
python scripts/03_train.py --epochs 3 --merge
python scripts/05_evaluate.py --verbose
```
