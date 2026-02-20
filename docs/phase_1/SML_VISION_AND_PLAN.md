# SML (Semantic Markup Language) — Vision, Architecture & Build Plan

**Version:** 1.0
**Date:** February 19, 2026
**Author:** Forrester
**Status:** Research & Planning Phase
**Companion Project:** CodeDNA (parallel track)

---

## Table of Contents

1. [The Vision: What is SML?](#1-the-vision)
2. [Why This Matters (The Problem SML Solves)](#2-why-this-matters)
3. [Architecture Overview](#3-architecture-overview)
4. [The SML Array Schema Design](#4-the-sml-array-schema-design)
5. [The Bible: Taxonomy & Knowledge Sources](#5-the-bible)
6. [Model Selection Decisions](#6-model-selection-decisions)
7. [Training Data Strategy](#7-training-data-strategy)
8. [The Fine-Tuning Plan](#8-the-fine-tuning-plan)
9. [Inference Flow (How It Works at Runtime)](#9-inference-flow)
10. [Evaluation & Benchmarking](#10-evaluation-and-benchmarking)
11. [Open Questions & Research Needed](#11-open-questions)
12. [Phase Roadmap](#12-phase-roadmap)
13. [Risk Assessment](#13-risk-assessment)

---

## 1. The Vision: What is SML?

### Core Concept

SML (Semantic Markup Language) is a numeric array-based representation language that encodes real-world meaning into compact, structured arrays. Unlike natural language (which is ambiguous) or traditional embeddings (which are opaque), SML arrays are:

- **Human-interpretable** — each array position has a defined semantic meaning
- **Compact** — a few numbers encode what would take sentences in natural language
- **Composable** — arrays can nest, chain, and reference each other
- **Grounded** — every value maps to a real-world concept via the "Bible" (ontology dictionary)

### The Key Insight

Current LLMs process text tokens, but those tokens are arbitrary symbols. The word "bank" activates fuzzy weight patterns that might mean a financial institution, a river bank, or a blood bank. SML pre-resolves this ambiguity BEFORE the model reasons about it.

Instead of asking the model to simultaneously:
1. Parse language
2. Resolve ambiguity
3. Retrieve relevant knowledge
4. Reason about the problem
5. Generate a response

SML splits this into:
1. **Encoder** handles parsing and disambiguation → produces SML arrays
2. **Model** receives pre-resolved, structured SML → reasons with loaded context
3. **Model** generates response grounded in the SML context

### The COT Integration (The Killer Feature)

The model's response format during both training and inference:

```text
<sml>
[Structured semantic arrays are LOADED here as pre-resolved context.
Through training, these arrays activate meaningful weight patterns —
the model learns that [3, 7, 2] MEANS "mammal, canine, domesticated"
and can use that grounded meaning in its reasoning.]
</sml>
<thinking>
[Reasoning occurs here, GROUNDED by the loaded SML context.
The model can reference SML concepts, chain logical steps,
and build on the structured foundation.]
</thinking>
<response>
[Final output to the user, informed by grounded reasoning.]
</response>
```

This is analogous to how experts think: a doctor doesn't re-derive anatomy from scratch for each patient — they have pre-loaded structured knowledge (the SML) that their reasoning builds upon.

---

## 2. Why This Matters (The Problem SML Solves)

### Current LLM Limitations
- **Hallucination**: Models confabulate because they lack grounded knowledge structures
- **Shallow reasoning**: Models pattern-match rather than truly understanding relationships
- **Inefficiency**: Small models waste capacity on disambiguation that could be pre-resolved
- **Opacity**: No way to inspect what the model "knows" or how it connects concepts

### What SML Changes
- **Grounded generation**: The model generates from structured facts, not fuzzy patterns
- **Transparent reasoning**: The SML block shows exactly what context the model is working with
- **Small model amplification**: A 3B model with SML can operate with the precision of a much larger model because ambiguity resolution is externalized
- **Composable knowledge**: New concepts can be added to the Bible without retraining

### Relationship to CodeDNA

CodeDNA and SML are complementary parallel tracks testing the same thesis at different depths:

| Aspect | CodeDNA | SML |
|--------|---------|-----|
| Approach | External scaffolding (no fine-tuning) | Internal cognition (fine-tuned into model) |
| Speed to results | Weeks (pure architecture) | Months (requires fine-tuning) |
| Depth of integration | Model never sees DNA directly | Model natively thinks in SML |
| Domain | Code generation (narrow) | General knowledge (broad) |
| Risk | Lower (no model modification) | Higher (could degrade model) |
| Potential ceiling | Limited by prompt engineering | Much higher if thesis holds |

If CodeDNA shows the "structured representation lift" is real, SML is the natural next step to embed that structure INTO the model.

---

## 3. Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        SML SYSTEM                            │
│                                                              │
│  ┌──────────┐     ┌─────────────┐     ┌──────────────────┐  │
│  │ ENCODER  │────▶│  SML BIBLE  │────▶│ FINE-TUNED LLM   │  │
│  │          │     │ (Ontology   │     │ (Qwen2.5-3B +    │  │
│  │ Converts │     │  Dictionary)│     │  SML LoRA)        │  │
│  │ input to │     │             │     │                   │  │
│  │ SML      │     │ ConceptNet  │     │ <sml>...</sml>    │  │
│  │ arrays   │     │ + WordNet   │     │ <thinking>...</>  │  │
│  │          │     │ + Wikidata  │     │ <response>...</>  │  │
│  └──────────┘     │ + Custom    │     └──────────────────┘  │
│       ▲           └─────────────┘            │               │
│       │                  ▲                   ▼               │
│  Raw Input          Bible Lookup        SML-Grounded         │
│  (text, later       & Encoding          Response             │
│   multimodal)                                                │
└─────────────────────────────────────────────────────────────┘
```

### Training Pipeline

```
Phase 1: Build SML Bible
  ConceptNet + WordNet + Wikidata → Unified Ontology → Numeric Array Mapping

Phase 2: Generate Training Data
  Large model (70B via Groq) takes existing datasets →
  Augments each example with SML arrays →
  Produces (input, sml_block, thinking, response) training tuples

Phase 3: Fine-Tune
  Qwen2.5-3B-Instruct + LoRA/QLoRA →
  Train on SML-augmented data →
  Model learns to consume and produce SML

Phase 4: Build/Train Encoder
  Separate small model or rule-based system →
  Converts raw text → SML arrays using the Bible
```

---

## 4. The SML Array Schema Design

### Design Principles

1. **Positional encoding**: Each position in an array has a fixed semantic role
2. **Numeric values**: All concepts map to integer IDs (not strings) for compactness
3. **Nestable**: Arrays can contain sub-arrays for complex structures
4. **Sparse-friendly**: Unused positions are 0 (null/not-applicable)
5. **Dictionary-backed**: Every non-zero value maps to a concept in the Bible

### Proposed Core Array Structure

After analyzing ConceptNet's relation types, WordNet's taxonomy, and Wikidata's property system, here is a proposed array schema. The data we find will refine this, but this is a strong starting structure:

#### Entity Descriptor Array (EDA) — Describes WHAT something is

```
Position  | Name              | Description                          | Example
----------|-------------------|--------------------------------------|--------
[0]       | domain            | Top-level domain ID                  | 1=physical, 2=abstract, 3=action, 4=property, 5=relation, 6=event
[1]       | category          | Primary category within domain       | Within physical: 1=living, 2=object, 3=substance, 4=place
[2]       | subcategory       | Refinement of category               | Within living: 1=human, 2=animal, 3=plant, 4=microorganism
[3]       | specificity       | Most specific classification          | Within animal: 1=mammal, 2=bird, 3=reptile, 4=fish, 5=insect
[4]       | instance_id       | Specific concept ID from Bible        | 4527 = "dog" (lookup in Bible dictionary)
[5]       | modifier_1        | Primary modifier/attribute            | Size: 1=tiny, 2=small, 3=medium, 4=large, 5=huge
[6]       | modifier_2        | Secondary modifier/attribute          | 0=none, or another property ID
[7]       | confidence        | How confident the encoding is (0-100) | 95 = very confident, 40 = ambiguous
```

**Example**: "a large golden retriever" → `[1, 1, 2, 1, 4527, 4, 892, 95]`
- domain=physical, category=living, subcategory=animal, specificity=mammal
- instance=4527("dog"), modifier_1=4(large), modifier_2=892("golden"), confidence=95

#### Relation Array (RA) — Describes HOW entities relate

```
Position  | Name              | Description                          | Example
----------|-------------------|--------------------------------------|--------
[0]       | relation_type     | Type of relationship                 | 1=IsA, 2=PartOf, 3=HasProperty, 4=UsedFor, 5=CapableOf, 6=AtLocation, 7=Causes, 8=HasPrerequisite, 9=MotivatedBy, 10=CreatedBy, ...
[1]       | subject_ref       | Index of subject entity in SML block | 0 = first entity described
[2]       | object_ref        | Index of object entity in SML block  | 1 = second entity described
[3]       | strength          | Relationship strength (0-100)        | 90 = strong/definite
[4]       | temporal          | Temporal qualifier                   | 0=atemporal, 1=past, 2=present, 3=future, 4=habitual
[5]       | negation          | 0=positive, 1=negated                | 1 = "is NOT"
```

**Example**: "The dog is sitting on the mat" →
- Entity 0: `[1, 1, 2, 1, 4527, 3, 0, 95]` (dog)
- Entity 1: `[1, 2, 0, 0, 8103, 2, 0, 90]` (mat - physical object)
- Relation: `[6, 0, 1, 85, 2, 0]` (AtLocation, dog→mat, strength=85, present, not negated)

#### Action/Event Array (AEA) — Describes WHAT HAPPENS

```
Position  | Name              | Description                          | Example
----------|-------------------|--------------------------------------|--------
[0]       | action_type       | Verb/action category ID              | From action taxonomy in Bible
[1]       | agent_ref         | Who/what performs the action          | Entity index
[2]       | patient_ref       | Who/what receives the action          | Entity index (0 if intransitive)
[3]       | instrument_ref    | Tool/means used                      | Entity index or 0
[4]       | manner            | How the action is performed           | Manner modifier ID
[5]       | aspect            | Temporal aspect                      | 1=ongoing, 2=completed, 3=habitual, 4=about-to
[6]       | intensity         | Force/degree (0-100)                 | 50=normal, 90=extreme
```

#### Composite SML Block

A full SML block for a sentence combines these arrays:

```
<sml>
entities: [[1,1,2,1,4527,4,892,95], [1,2,0,0,8103,2,0,90]]
relations: [[6,0,1,85,2,0]]
actions: [[2341,0,0,0,0,1,30]]
context: [2,0,0,3,0]
</sml>
```

The `context` array provides scene-level metadata:
- [0] = setting (1=indoor, 2=outdoor, 3=abstract, 4=digital)
- [1] = formality (0=neutral, 1=casual, 2=formal, 3=technical)
- [2] = domain (0=general, 1=science, 2=social, 3=technical, ...)
- [3] = time_reference (0=none, 1=historical, 2=current, 3=future)
- [4] = sentiment (0=neutral, 1=positive, 2=negative, 3=mixed)

### Array Position Count Summary

| Array Type | Positions | Purpose |
|------------|-----------|---------|
| Entity Descriptor | 8 | What something IS |
| Relation | 6 | How things RELATE |
| Action/Event | 7 | What HAPPENS |
| Context | 5 | Scene-level metadata |

**Total positions per concept:** 8 (entity) + up to 6 (relation) + up to 7 (action) + 5 (context) = **~26 numbers** can describe a rich semantic scene.

For comparison: the sentence "A large golden retriever is sitting on a red mat in the park" takes ~15 tokens in a tokenizer. The SML representation takes ~30-40 numbers but encodes MUCH more semantic information (taxonomy, relationships, confidence, temporal aspect, etc.) that the model would otherwise need to infer.

### What Dictates the Final Schema

The schema above is a proposal. The actual schema will be refined by:

1. **ConceptNet's relation types** (34 relations) → directly map to relation_type values
2. **WordNet's taxonomy depth** → determines how many levels domain/category/subcategory needs
3. **Coverage analysis** → run a sample of training data through the schema, count how often we hit "can't encode this" → add positions as needed
4. **Token efficiency analysis** → how many tokens does the SML representation cost vs. the information gained?
5. **Model learning analysis** → during early fine-tuning, which positions does the model actually attend to? Prune unused ones.

---

## 5. The Bible: Taxonomy & Knowledge Sources

### Tier 1: Primary Sources (Use Directly)

#### ConceptNet 5.7 (THE Foundation)
- **What**: Open multilingual knowledge graph with ~34 relation types and millions of edges
- **Size**: ~8M edges, ~3M nodes across 78 languages
- **Relations**: IsA, HasA, PartOf, UsedFor, CapableOf, AtLocation, Causes, HasPrerequisite, MotivatedBy, CreatedBy, DefinedAs, SymbolOf, MadeOf, HasProperty, HasContext, and ~20 more
- **Format**: TSV/JSON, freely downloadable (CC BY-SA 4.0)
- **Why perfect for SML**: Already structured as subject-relation-object triples. The relation types map directly to our Relation Array schema. Concepts are normalized and cross-linked.
- **ConceptNet Numberbatch**: Pre-computed numeric embeddings for concepts — these can bootstrap our concept ID assignments
- **Download**: https://github.com/commonsense/conceptnet5 (CSV dumps available)

#### WordNet 3.1
- **What**: Lexical database organizing English words into synsets (synonym groups) with hierarchical taxonomy
- **Size**: ~117,000 synsets, ~155,000 words
- **Structure**: Nouns organized in a deep IS-A hierarchy (e.g., dog → canine → carnivore → mammal → vertebrate → animal → organism → entity)
- **Why perfect for SML**: Provides the taxonomy backbone for our domain/category/subcategory/specificity positions. WordNet's hypernym chains ARE the array positions.
- **Access**: NLTK (`from nltk.corpus import wordnet`), or direct download

#### Wikidata
- **What**: Free, structured knowledge base with 100M+ items
- **Size**: Massive — 100M+ entities, 1.5B+ statements
- **Why useful**: Named entity coverage, property definitions, cross-links to everything
- **Caution**: Too large to use wholesale. Cherry-pick: property definitions (P-values) for our modifier system, entity types for our category system
- **Access**: SPARQL endpoint, JSON dumps

### Tier 2: Supplementary Sources

| Source | What It Provides | Size | Use For |
|--------|-----------------|------|---------|
| **DBpedia** | Structured Wikipedia extracts | 2B+ triples | Named entity enrichment |
| **YAGO** | High-accuracy knowledge from Wikipedia+WordNet | 50M+ facts | Clean taxonomy data |
| **schema.org** | Shared vocabulary for web data | ~800 types | Property standardization |
| **FrameNet** | Semantic frames for actions/events | 1,200+ frames | Action/Event array design |
| **VerbNet** | Verb classifications and thematic roles | 6,000+ verbs | Agent/patient/instrument roles |
| **ATOMIC** | Commonsense reasoning graphs (if-then) | 877K inferences | Causal reasoning chains |
| **Visual Genome** | Scene graph annotations | 108K images, 2.3M relationships | Future: vision SML training |

### Tier 3: Domain-Specific (Phase 2+)

- **UMLS** (medical ontology): 4M+ concepts — for medical domain SML
- **FIBO** (financial ontology): For financial reasoning
- **Gene Ontology**: For bioinformatics

### Bible Construction Pipeline

```
Step 1: EXTRACT taxonomies from WordNet
  └── Build domain → category → subcategory → specificity hierarchy
  └── Assign numeric IDs to each level
  └── Result: ~50-100 top-level categories, ~500-2000 subcategories

Step 2: EXTRACT concepts from ConceptNet
  └── Map each concept to its WordNet synset (already linked!)
  └── Assign concept instance_id values
  └── Result: ~100K most common concept IDs

Step 3: EXTRACT relations from ConceptNet
  └── Map ConceptNet's 34 relation types to our Relation Array relation_type positions
  └── Validate coverage against sample sentences
  └── Result: Relation type mapping table

Step 4: ENRICH with Wikidata properties
  └── Add modifier definitions for common properties (color, size, material, etc.)
  └── Add temporal/geographic qualifiers
  └── Result: Modifier lookup tables

Step 5: COMPILE into SML Bible
  └── Single JSON/SQLite file mapping:
      - concept_name → instance_id
      - instance_id → [domain, category, subcategory, specificity]
      - relation_name → relation_type_id
      - property_name → modifier_id
  └── Include reverse lookups for decoder
  └── Include frequency statistics for prioritization

Step 6: VALIDATE
  └── Take 1000 random English sentences
  └── Attempt to encode each into SML using the Bible
  └── Measure: % of concepts found, % of relations covered, % of failures
  └── Target: >85% coverage on common language
```

### Estimated Bible Size
- **Concept IDs**: ~100,000 entries (covering most common English words + key entities)
- **Taxonomy levels**: ~4,000 unique category paths
- **Relation types**: ~40 (ConceptNet's 34 + custom additions)
- **Modifier values**: ~500 common modifiers (size, color, material, speed, etc.)
- **Total file size**: ~50-100MB as JSON, ~20-40MB as SQLite

---

## 6. Model Selection Decisions

### Decision: Text-Only Qwen2.5-3B-Instruct (NOT VL)

**Recommendation: Use Qwen2.5-3B-Instruct (text-only) for Phase 1**

| Factor | Qwen2.5-3B (Text) | Qwen2.5-VL-3B (Vision) |
|--------|-------------------|------------------------|
| Parameter efficiency | All 3B params for language | ~30% params for ViT (vision), ~70% for language |
| Text performance | Strong on math, coding, structured output | Slightly lower on pure text tasks |
| Fine-tuning simplicity | Standard LoRA, well-documented | VL fine-tuning more complex (multi-modal alignment) |
| VRAM for fine-tuning | ~6-8GB with QLoRA | ~10-12GB with QLoRA (ViT overhead) |
| SML relevance | SML is text/numeric — no images needed | Vision capability unused in Phase 1 |
| Training data format | Simple text tuples | Would need dummy image inputs |
| Catastrophic forgetting risk | Lower (modifying one modality) | Higher (multi-modal balance fragile) |
| Structured output | Specifically improved in Qwen2.5 series | Optimized more for visual grounding |
| License | Qwen Research License | Apache 2.0 |

**Bottom line**: Using VL-3B for a text-only SML task is like buying a pickup truck to deliver letters. Save vision for Phase 2 when we extend SML to images.

**Alternative consideration**: Qwen3-4B is now available (as of May 2025) with hybrid thinking mode. Worth evaluating if:
- Its thinking mode aligns naturally with our `<thinking>` block
- Performance gains justify the switch
- But stick with Qwen2.5-3B for initial PoC since tooling is more mature

### Model for Generating Training Data

**Recommendation: Use Groq with llama-3.3-70b-versatile OR qwen2.5-72b**

Why:
- We need a LARGE model to produce high-quality SML annotations
- The training data generator must understand the SML schema perfectly
- 70B+ models can follow complex structured output instructions reliably
- Groq provides fast inference for the large volume we need (5K-50K examples)
- Local 7B models will make too many SML encoding errors

Pipeline:
1. Feed 70B model: the SML Bible + schema spec + a raw training example
2. Model outputs: properly formatted (input, sml_block, thinking, response) tuple
3. Validate: check SML arrays against Bible (deterministic validation)
4. Reject bad samples, collect good ones

### Separate Encoder Model

**For Phase 1**: The encoder can be rule-based (not ML):
- Parse input text with spaCy NER + dependency parsing
- Look up entities in Bible dictionary
- Map relations using dependency parse structure
- This is deterministic and debuggable

**For Phase 2**: Fine-tune a separate small model (Qwen2.5-0.5B or 1.5B) as an SML encoder:
- Train on (raw_text → sml_arrays) pairs from Phase 1
- Much smaller model since encoding is simpler than generation

---

## 7. Training Data Strategy

### The Core Question: How Much Data?

This is the "billions of examples" fear. Let's address it head-on.

**You are NOT replacing the model's knowledge.** Qwen2.5-3B already knows what a dog is, what "sitting" means, what spatial relationships are. It learned this from 18 TRILLION tokens of pre-training. You cannot and should not try to replicate that.

**You ARE teaching it a new output format.** Specifically, you're teaching it:
1. How to read SML arrays (what `[1,1,2,1,4527,4,892,95]` means)
2. How to reference SML context in its reasoning
3. How to produce responses grounded in SML-loaded context

This is a FORMAT task, not a KNOWLEDGE task. The research is clear on format tasks:

- **Minimum viable**: ~1,000 high-quality examples can teach a new output format
- **Solid performance**: ~5,000-10,000 examples for reliable structured output
- **Strong generalization**: ~20,000-50,000 examples for robust performance across diverse inputs
- **Diminishing returns**: Beyond 50K, improvements are marginal for format learning

**However**: The diversity of SML concepts IS a knowledge task. The model needs to see enough DIFFERENT SML arrays to learn what each position means. This requires:
- Examples covering all major domain values (1-6)
- Examples covering all major relation types (~34)
- Examples covering diverse concept IDs (~top 10,000 concepts)
- Examples with various nesting depths

**Recommended dataset size: 20,000-30,000 high-quality examples** for Phase 1, with plans to scale to 100K if results are promising.

### Base Datasets to Augment with SML

We need diverse, high-quality instruction-following datasets that cover a broad range of topics. The SML augmentation adds the `<sml>` block — it does NOT replace the original training format.

#### Primary Dataset Candidates

| Dataset | Size | Why Use It | SML Fit |
|---------|------|-----------|---------|
| **OpenHermes 2.5** | 1M samples | Diverse, high-quality instruction-response pairs | Excellent — broad topic coverage |
| **SlimOrca** | 518K samples | Cleaned, deduplicated subset of OpenOrca | Good — curated quality |
| **Dolly 15k** | 15K samples | Human-written instruction-response pairs | Good for initial testing |
| **MMLU (few-shot)** | 14K questions | Tests broad knowledge across 57 subjects | Great for knowledge grounding validation |
| **Alpaca GPT-4** | 52K samples | GPT-4 generated instruction data | Good diversity, decent quality |
| **WizardLM Evol-Instruct** | 70K samples | Complexity-evolved instructions | Tests SML with increasingly complex reasoning |
| **ATOMIC 2020** | 877K tuples | If-then commonsense reasoning | Perfect for relation/causation SML training |
| **ShareGPT** | ~90K conversations | Real user conversations with AI | Tests SML with natural conversational patterns |

#### Augmentation Strategy: Subset + Quality

We do NOT augment all 1M+ samples. Instead:

```
Step 1: SAMPLE — Take 30,000 diverse examples from OpenHermes + SlimOrca + Dolly
  - Ensure coverage across topics: science, social, technical, creative, factual
  - Ensure coverage across task types: QA, reasoning, classification, generation
  - Stratified sampling by topic + complexity

Step 2: AUGMENT — Use 70B model (Groq) to generate SML for each example
  - Input: raw example + SML Bible + SML schema spec
  - Output: SML-annotated version with <sml>, <thinking>, <response> blocks
  - Batch process through Groq API (~10-30 examples per minute)

Step 3: VALIDATE — Deterministic checks
  - All concept IDs exist in Bible ✓
  - Array positions match schema (correct length, valid ranges) ✓
  - Relations reference valid entity indices ✓
  - No hallucinated concept IDs ✓
  - Reject ~10-20% expected failure rate

Step 4: CLEAN — Remove low-quality samples
  - Encoder confidence below 50 → remove
  - SML block doesn't match the response content → remove
  - Target: ~25,000 clean training examples

Step 5: SPLIT — 90% train / 5% validation / 5% test
  - Ensure topic distribution is balanced across splits
```

### Training Data Format

Each training example follows the ChatML format with SML blocks:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant that uses Semantic Markup Language (SML) to ground your reasoning. When provided with SML context, use it to inform your thinking and ensure your response is accurate and well-grounded."
    },
    {
      "role": "user",
      "content": "What is the largest mammal on Earth?"
    },
    {
      "role": "assistant",
      "content": "<sml>\nentities: [[1,1,2,1,8842,5,0,98]]\nrelations: [[3,0,-1,99,2,0]]\ncontext: [0,0,1,2,0]\n</sml>\n<thinking>\nThe query asks about the largest mammal. From the SML context, entity 8842 maps to \"blue whale\" in the domain of living organisms, specifically mammals. The HasProperty relation confirms the \"largest\" attribute with 99% confidence. This is current, well-established biological knowledge.\n</thinking>\n<response>\nThe largest mammal on Earth is the blue whale (Balaenoptera musculus). Adult blue whales can reach lengths of up to 100 feet (30 meters) and weigh as much as 200 tons, making them not just the largest mammal but the largest animal known to have ever existed.\n</response>"
    }
  ]
}
```

---

## 8. The Fine-Tuning Plan

### Method: QLoRA (Quantized Low-Rank Adaptation)

Why QLoRA:
- Fits on RTX 4090 (24GB VRAM) easily
- Preserves base model knowledge (critical — we don't want to degrade the model)
- Fast iteration (~2-4 hours for a training run at 20K examples)
- Well-supported by Unsloth framework

### Fine-Tuning Configuration

```python
# Using Unsloth for efficient QLoRA training
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length=4096,  # SML blocks add length
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                    # LoRA rank — higher for learning new format
    lora_alpha=128,          # Scaling factor
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

### Training Hyperparameters

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./sml-qwen-3b",
    num_train_epochs=3,              # 3 epochs for format learning
    per_device_train_batch_size=4,   # Fits in 24GB with 4bit
    gradient_accumulation_steps=4,   # Effective batch size = 16
    learning_rate=2e-4,              # Standard for QLoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    fp16=True,                       # or bf16 if supported
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    max_seq_length=4096,
)
```

### Training Phases

#### Phase A: Format Learning (Week 1)
- **Data**: 5,000 examples with simple SML (1-2 entities, 1 relation)
- **Goal**: Model learns the `<sml>/<thinking>/<response>` format
- **Validation**: Does the model produce valid SML-structured output?
- **Expected time**: ~1-2 hours on RTX 4090

#### Phase B: Concept Coverage (Week 2)
- **Data**: 15,000 examples with diverse SML (3-5 entities, multiple relations)
- **Goal**: Model learns to reference SML concepts in reasoning
- **Validation**: Does the thinking block correctly reference SML entities?
- **Expected time**: ~3-4 hours on RTX 4090

#### Phase C: Grounding Quality (Week 3)
- **Data**: Full 25,000 examples including complex reasoning chains
- **Goal**: Model produces grounded, accurate responses using SML
- **Validation**: Compare response accuracy with vs. without SML grounding
- **Expected time**: ~4-6 hours on RTX 4090

### Hardware Requirements

| Setting | RTX 4090 (24GB) | M4 Mac (24GB) |
|---------|-----------------|----------------|
| QLoRA 4-bit | ✅ ~12GB VRAM | ✅ ~12GB unified |
| Batch size | 4 | 2-4 |
| Training time (25K samples, 3 epochs) | ~4-6 hours | ~8-12 hours |
| Recommended framework | Unsloth | Unsloth (MLX backend available) |

---

## 9. Inference Flow (How It Works at Runtime)

### Step-by-Step Inference

```
USER INPUT: "Why do cats purr?"

Step 1: ENCODER (rule-based or small model)
  ├── Parse: subject="cats", action="purr", question_type="why/causation"
  ├── Bible lookup: cat → instance_id=3891, [1,1,2,1,3891,3,0,95]
  ├── Bible lookup: purr → action_id=7234, behavior category
  ├── Relation: HasCapability(cat, purr), Causes(purr, ?)
  └── Output SML block

Step 2: COMPOSE PROMPT
  System: "You are an SML-grounded assistant..."
  User: "Why do cats purr?"
  + Injected SML block:
    <sml>
    entities: [[1,1,2,1,3891,3,0,95], [3,5,0,0,7234,0,0,90]]
    relations: [[5,0,1,95,4,0], [7,1,-1,70,4,0]]
    context: [0,0,1,2,0]
    </sml>

Step 3: MODEL GENERATES (fine-tuned Qwen2.5-3B)
  <thinking>
  The query asks about the cause of purring in cats (entity 3891).
  SML indicates this is a biological behavior (action 7234) with
  a CapableOf relation at 95% confidence. The Causes relation from
  purring outward has 70% confidence — multiple hypotheses exist.
  Key known causes include communication, self-healing, and comfort.
  </thinking>
  <response>
  Cats purr for several reasons. The primary theories include...
  [grounded, accurate response]
  </response>

Step 4: RETURN response block to user
```

### Performance Characteristics

| Step | Time (RTX 4090) | Time (M4 Mac) |
|------|----------------|----------------|
| Encoding (rule-based) | ~5-20ms | ~5-20ms |
| Encoding (small model) | ~50-100ms | ~100-200ms |
| Model generation (3B, 4-bit) | ~200-800ms | ~400-1500ms |
| Total pipeline | ~250-900ms | ~500-1700ms |

Fast enough for interactive use.

---

## 10. Evaluation & Benchmarking

### What to Measure

#### A. Format Compliance
- Does the model produce valid `<sml>/<thinking>/<response>` blocks?
- Are SML arrays valid (correct positions, values in range)?
- Target: >95% format compliance

#### B. Grounding Accuracy
- When given SML context, does the response align with it?
- Does the model contradict its own SML block? (Should NEVER happen)
- Target: >90% alignment between SML and response

#### C. Quality vs. Baseline
- Compare SML-augmented model vs. base Qwen2.5-3B-Instruct on:
  - **MMLU** (broad knowledge)
  - **TruthfulQA** (hallucination resistance — SML should help here!)
  - **ARC-Challenge** (reasoning)
  - **HellaSwag** (commonsense)
- Critical: SML model should NOT degrade on these. Improvement is bonus.

#### D. Ablation Studies
- Model with SML block vs. same model without SML block
- Model with correct SML vs. model with intentionally wrong SML (should produce worse answers)
- Full SML vs. partial SML (only entities, no relations) — what matters most?

#### E. Hallucination Resistance (The Key Metric)
- Design 100 questions where the SML bible contains the correct answer
- Design 100 questions where the SML intentionally provides NO relevant context
- Measure: Does the model refuse to confabulate when SML is empty/irrelevant?
- This is the core value proposition — SML should make the model MORE honest

### Benchmark Datasets

| Benchmark | Purpose | Expected SML Effect |
|-----------|---------|-------------------|
| MMLU | General knowledge | Neutral to slight positive |
| TruthfulQA | Hallucination resistance | **Significant positive** (core thesis) |
| ARC-Challenge | Scientific reasoning | Positive (SML provides structured facts) |
| HellaSwag | Commonsense completion | Positive (SML encodes commonsense) |
| Custom SML Grounding Test | SML-specific eval | Must pass >90% |
| BBH (Big Bench Hard) | Complex reasoning | Positive if SML improves reasoning chain |

---

## 11. Open Questions & Research Needed

### High Priority (Must Resolve Before Building)

1. **Optimal array width**: Is 8 positions enough for Entity Descriptors, or do we need 12? Need coverage analysis against a corpus.

2. **Concept ID assignment strategy**: Do we use ConceptNet Numberbatch embeddings directly? Or assign new sequential IDs based on frequency? Numberbatch gives semantic similarity for free, but adds ~300D of embedding per concept. Sequential IDs are simpler but lose inter-concept relationships.

3. **SML token budget**: How many tokens does a typical SML block consume? If average SML block is 200 tokens, that's significant for a 4096-token context window. May need to optimize encoding to be more compact.

4. **Graceful degradation**: When the encoder can't find a concept in the Bible, what happens? Options: (a) pass-through as text, (b) use nearest match, (c) mark as unknown. Need to decide and train for this case.

5. **Training data pipeline latency**: At Groq rates (~30 req/min on free tier), 30K examples could take ~17 hours of continuous generation. Need to plan: use paid tier? Parallelize across accounts? Generate in batches over days?

### Medium Priority (Can Resolve During Building)

6. **Encoder architecture**: Pure rule-based vs. small fine-tuned model vs. hybrid. Start rule-based, measure quality, decide if ML encoder is needed.

7. **Merging encoder with LLM**: Your diagram asks about this. For PoC: definitely keep separate. For production: investigate LoRA adapter stacking or multi-task training to unify.

8. **Multi-turn conversations**: How does SML work across conversation turns? Does each turn get its own SML block? Do we maintain a running SML state? Need to design this.

9. **SML version evolution**: When we add new concepts to the Bible or change array positions, how do we handle backward compatibility? Need a versioning scheme.

### Low Priority (Phase 2+)

10. **Vision encoder for SML**: When we extend to images, does the ViT produce SML directly, or does it produce features that a separate encoder converts to SML?

11. **Audio encoder for SML**: Speech → text → SML is straightforward. But can we encode prosody, emotion, speaker identity directly in SML?

12. **SML compression**: Can we learn a more compact SML representation through training? Auto-encode the arrays into even fewer dimensions?

---

## 12. Phase Roadmap

### Phase 1: Foundation (Weeks 1-4)

| Week | Tasks | Deliverables |
|------|-------|-------------|
| 1 | Build SML Bible from ConceptNet + WordNet | Bible v1 (JSON/SQLite), ~100K concepts |
| 1 | Finalize array schema (test against 1000 sentences) | Schema spec document |
| 2 | Build rule-based encoder (spaCy + Bible lookup) | Encoder module |
| 2 | Begin training data generation (Groq pipeline) | First 5K examples |
| 3 | Complete training data generation + validation | 25K validated examples |
| 3 | Phase A fine-tune: format learning | Model checkpoint A |
| 4 | Phase B+C fine-tune: concept coverage + grounding | Model checkpoint C |
| 4 | Run initial benchmarks (MMLU, TruthfulQA, custom) | Benchmark results v1 |

### Phase 2: Refinement (Weeks 5-8)

- Analyze Phase 1 results, identify weak points
- Expand Bible with domain-specific knowledge
- Scale training data to 50-100K if Phase 1 shows promise
- Build ML-based encoder (fine-tune Qwen2.5-0.5B or 1.5B)
- Design multi-turn conversation SML protocol
- Cross-reference results with CodeDNA track

### Phase 3: Vision Extension (Weeks 9-12)

- Integrate Qwen2.5-VL-3B as vision encoder
- Train vision-to-SML conversion (using Visual Genome dataset)
- Multi-modal SML: text + image inputs producing unified SML
- Extended benchmarks on visual QA tasks

### Phase 4: Publication & Integration (Weeks 13+)

- Write up findings
- Open-source the Bible and training pipeline
- Integrate SML encoder into FoFoBrain as a pre-processing layer
- Explore SML as shared representation between FoFoBrain agents

---

## 13. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| SML arrays don't activate meaningful weight patterns after fine-tuning | Fatal | Medium | Start with Phase A (format only) to verify model can learn the format before investing in full data generation |
| Bible coverage insufficient for diverse topics | High | Medium | Start with ConceptNet's broad coverage (~8M edges). Add "unknown" fallback. Measure coverage rate on diverse corpus. |
| Training data quality too low from Groq generation | High | Medium | Deterministic validation rejects bad samples. Use 70B+ model for generation. Human-spot-check random samples. |
| Fine-tuning degrades base model performance | High | Low | QLoRA preserves most base weights. Run MMLU before/after. If >2% degradation, reduce LoRA rank or training epochs. |
| SML blocks consume too many context tokens | Medium | Medium | Optimize encoding: use compact format, abbreviate array names, measure actual token cost early. Consider binary encoding. |
| Encoder (rule-based) too brittle for diverse inputs | Medium | High | Expected — rule-based is intentionally Phase 1 only. Measure error rate, use as training signal for Phase 2 ML encoder. |
| Cannot generate 25K training examples in reasonable time | Medium | Low | Groq paid tier is cheap. Can also use Claude/GPT-4 API as alternative. Parallelize generation scripts. |
| The model memorizes SML patterns without actually understanding them | High | Medium | Ablation study: give wrong SML and measure if output degrades. If it doesn't degrade, model isn't using SML — it's memorizing. |

### The "From Scratch" Fear

**Do we need to train from scratch?** No. Here's the math:

- Qwen2.5-3B was pre-trained on **18 trillion tokens**
- Our SML fine-tune: ~25K examples × ~500 tokens avg = **12.5 million tokens**
- That's 0.00007% of the pre-training data
- We are adding a thin layer of format understanding, not replacing knowledge
- LoRA rank 64 touches ~2-3% of total parameters
- This is equivalent to teaching someone who speaks English fluently how to fill out a specific form — you're not teaching them English

**If the model's existing knowledge is good enough** (and 3B Qwen2.5 scores 65+ on MMLU, so it is), then SML just gives that knowledge a structured way to express itself. The model already "knows" what a dog is — SML teaches it that `[1,1,2,1,4527,4,892,95]` is how to formally express that knowledge.

---

## Appendix A: ConceptNet Relation Types → SML Relation Array Mapping

| ConceptNet Relation | SML relation_type ID | Description |
|--------------------|---------------------|-------------|
| IsA | 1 | Taxonomy: X is a type of Y |
| PartOf | 2 | Meronymy: X is part of Y |
| HasA | 3 | X has Y as a part or feature |
| HasProperty | 4 | X has property Y |
| CapableOf | 5 | X is capable of doing Y |
| AtLocation | 6 | X is typically found at Y |
| Causes | 7 | X causes Y to happen |
| HasPrerequisite | 8 | X requires Y |
| HasFirstSubevent | 9 | First step of X is Y |
| HasLastSubevent | 10 | Last step of X is Y |
| MotivatedByGoal | 11 | X is done to achieve Y |
| UsedFor | 12 | X is used for Y |
| CreatedBy | 13 | X is created by process Y |
| DefinedAs | 14 | X is defined as Y |
| SymbolOf | 15 | X symbolizes Y |
| MadeOf | 16 | X is made of Y |
| ReceivesAction | 17 | X receives action Y |
| Desires | 18 | X wants Y |
| CausesDesire | 19 | X makes you want Y |
| HasContext | 20 | X is used in context Y |
| SimilarTo | 21 | X is similar to Y |
| Antonym | 22 | X is opposite of Y |
| DerivedFrom | 23 | X derives from Y |
| RelatedTo | 24 | General relationship |
| FormOf | 25 | X is a form of Y |
| EtymologicallyRelatedTo | 26 | Word origin relation |
| Synonym | 27 | X means the same as Y |
| MannerOf | 28 | X is a way of doing Y |
| LocatedNear | 29 | X is near Y |
| HasContext | 30 | X is used in domain Y |
| dbpedia/genre | 31 | Genre classification |
| dbpedia/occupation | 32 | Occupation |
| dbpedia/language | 33 | Language |
| dbpedia/capital | 34 | Capital city |
| (custom additions) | 35-50 | Reserved for project-specific |

---

## Appendix B: Comparison of Knowledge Sources for SML Bible

| Source | Concepts | Relations | Structure Depth | Open License | SML Fit |
|--------|----------|-----------|----------------|-------------|---------|
| **ConceptNet 5.7** | ~3M nodes | ~8M edges, 34 types | Flat graph | CC BY-SA 4.0 | ★★★★★ |
| **WordNet 3.1** | ~117K synsets | Hypernym/hyponym hierarchy | Deep tree (up to 20 levels) | Open | ★★★★★ |
| **Wikidata** | ~100M items | ~1.5B statements | Flat + sparse hierarchy | CC0 | ★★★☆☆ |
| **YAGO 4** | ~64M entities | Rich typing | Moderate | CC BY-SA | ★★★★☆ |
| **FrameNet** | ~1200 frames | Agent/patient/instrument roles | Moderate | Custom/academic | ★★★★☆ |
| **VerbNet** | ~6000 verbs | Thematic roles | Moderate | Open | ★★★★☆ |
| **ATOMIC 2020** | ~877K tuples | If-then commonsense | Flat (inference chains) | CC BY | ★★★★☆ |
| **Visual Genome** | ~108K images | 2.3M relationships | Scene graphs | CC BY 4.0 | ★★★☆☆ (Phase 2) |
| **schema.org** | ~800 types | Property definitions | Moderate hierarchy | CC BY-SA 3.0 | ★★★☆☆ |

---

*End of SML Vision, Architecture & Build Plan Document*
