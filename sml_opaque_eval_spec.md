# SML Opaque Entity Evaluation Suite — Implementation Spec

## PURPOSE

Build a custom evaluation task for EleutherAI's `lm-evaluation-harness` that tests whether an SML-trained model can perform **genuine relational reasoning** from SML structure alone — without help from semantic prefix extraction.

This is THE decisive experiment. If the fine-tuned model scores significantly better than the base model on these questions, SML provides real reasoning augmentation. If both models fail equally, the approach needs fundamental changes before more training data investment is justified.

## THE CORE IDEA

Standard benchmarks (ARC, HellaSwag, etc.) test things the model already knows. SML injection on those benchmarks just adds overhead. We need questions that can ONLY be answered by reading and following the SML relational structure.

To do this, we use **completely opaque entity tokens** — no semantic prefixes to extract. Instead of `penguin_8847` (where the model reads "penguin" from the subword tokens), we use `ent_8847` or `X1`, `X2`, etc. The model has ZERO pretrained knowledge about these entities. The ONLY source of information is the SML block's relational structure.

## WHAT TO BUILD

### 1. Dataset: `sml_opaque_reasoning.jsonl`

Generate **100 evaluation examples** as a JSONL file. Each example is a multiple-choice question (4 choices: A, B, C, D) with exactly one correct answer that can be determined ONLY by following the SML relations.

### 2. Task Config: `sml_opaque_reasoning.yaml`

A YAML task config for lm-evaluation-harness that loads the JSONL and runs the evaluation.

### 3. Runner Script: `run_sml_eval.py`

A convenience script that runs the eval against both the fine-tuned model and vanilla Qwen3-4B, then compares results.

---

## SML FORMAT REFERENCE

The model was trained on SML blocks in the user message with this format:

```
<sml>
E(domain|category|subcategory|specificity|anchor_token|modifier|temporal|confidence)
E(domain|category|subcategory|specificity|anchor_token|modifier|temporal|confidence)
R(RelationType|source_idx|target_idx|weight|temporal|negation)
</sml>
```

**Entity fields:**
- domain, category, subcategory, specificity: integers (use 0 for all in this eval — we're testing relation reasoning, not domain classification)
- anchor_token: the entity identifier (THIS IS WHAT WE MAKE OPAQUE)
- modifier: optional modifier or 0
- temporal: 0
- confidence: float 0.0-1.0

**Relation fields:**
- RelationType: one of IsA, HasA, PartOf, CapableOf, UsedFor, AtLocation, HasProperty, Causes, HasPrerequisite, RelatedTo, Antonym, Synonym, MadeOf, DerivedFrom, MannerOf, MotivatedByGoal, SimilarTo, FormOf, NOT_CapableOf, NOT_HasProperty (NOT_ prefix = negation)
- source_idx, target_idx: integer indices referring to entity order (0-based)
- weight: float 0.0-1.0 (confidence/strength)
- temporal: 0
- negation: 0 (use NOT_ prefix in relation type instead)

**CRITICAL: Opaque anchor tokens.** Use tokens like `X0`, `X1`, `X2` or `ent_001`, `ent_002` etc. Do NOT use semantic words like `dog`, `penguin`, `car`. The whole point is that the model cannot use pretrained knowledge — it MUST read the SML.

---

## QUESTION CATEGORIES (100 total)

### Category 1: Simple Property Lookup (20 questions)

Test whether the model can read a single relation and answer a direct question.

**Template:**
```
<sml>
E(0|0|0|0|X0|0|0|0.9)
E(0|0|0|0|X1|0|0|0.9)
E(0|0|0|0|X2|0|0|0.9)
E(0|0|0|0|X3|0|0|0.9)
R(HasProperty|0|1|0.85|0|0)
</sml>

According to the SML data, which property does X0 have?
A) X1
B) X2
C) X3
D) None of the above

Answer: A
```

Vary the relation types: IsA, HasA, PartOf, AtLocation, CapableOf, UsedFor, HasProperty.
Vary which entity index is the correct answer.
Vary the number of distractor entities (2-4 distractors).
Use different confidence weights (0.3-1.0).

### Category 2: Negation Reasoning (15 questions)

Test whether the model correctly interprets NOT_ relations.

**Template:**
```
<sml>
E(0|0|0|0|X0|0|0|0.9)
E(0|0|0|0|X1|0|0|0.9)
E(0|0|0|0|X2|0|0|0.9)
R(CapableOf|0|1|0.9|0|0)
R(NOT_CapableOf|0|2|0.9|0|0)
</sml>

According to the SML data, what is X0 NOT capable of?
A) X1
B) X2
C) Both X1 and X2
D) Neither

Answer: B
```

Mix questions asking what something CAN do vs what it CANNOT do.
Include cases with both positive and negative relations to the same entity type.
Include cases where the answer is "None" (no NOT_ relation exists).

### Category 3: Relation Chain / Multi-Hop (20 questions)

Test whether the model can follow a chain of 2-3 relations to reach a conclusion.

**Template:**
```
<sml>
E(0|0|0|0|X0|0|0|0.9)
E(0|0|0|0|X1|0|0|0.9)
E(0|0|0|0|X2|0|0|0.9)
R(IsA|0|1|0.85|0|0)
R(CapableOf|1|2|0.90|0|0)
</sml>

X0 is a type of X1. X1 is capable of X2. Based on the SML data, can X0 do X2?
A) Yes, because X0 inherits capabilities from X1
B) No, X0 has no capabilities listed
C) Only X1 can do X2
D) Insufficient information

Answer: A
```

Chain types to test:
- IsA inheritance: If X0 IsA X1, and X1 CapableOf X2, then X0 can do X2
- Location transitivity: X0 AtLocation X1, X1 PartOf X2, so X0 is within X2
- Prerequisite chains: X0 HasPrerequisite X1, X1 HasPrerequisite X2, so X0 requires X2
- Causal chains: X0 Causes X1, X1 Causes X2, so X0 indirectly causes X2

Include 2-hop and 3-hop chains. Include distractor entities that are NOT part of the chain.

### Category 4: Weight Comparison (15 questions)

Test whether the model can compare relation weights to determine strength.

**Template:**
```
<sml>
E(0|0|0|0|X0|0|0|0.9)
E(0|0|0|0|X1|0|0|0.9)
E(0|0|0|0|X2|0|0|0.9)
R(RelatedTo|0|1|0.85|0|0)
R(RelatedTo|0|2|0.25|0|0)
</sml>

According to the SML data, is X0 more strongly related to X1 or X2?
A) X1 (weight 0.85 vs 0.25)
B) X2 (weight 0.25 vs 0.85)
C) They are equally related
D) X0 is not related to either

Answer: A
```

Vary the gap between weights (large gap = easy, small gap = hard).
Include some equal-weight cases where C is correct.
Include cases with different relation types to same target.

### Category 5: Counting and Structure (10 questions)

Test whether the model can count entities, relations, or identify structural properties.

**Template:**
```
<sml>
E(0|0|0|0|X0|0|0|0.9)
E(0|0|0|0|X1|0|0|0.9)
E(0|0|0|0|X2|0|0|0.9)
R(IsA|0|1|0.8|0|0)
R(HasA|0|2|0.7|0|0)
R(CapableOf|1|2|0.6|0|0)
</sml>

How many relations involve X0 as a source?
A) 1
B) 2
C) 3
D) 0

Answer: B
```

Questions about: number of entities, number of relations, which entity has the most connections, which entity is isolated (no relations), etc.

### Category 6: Composite Reasoning (20 questions)

Harder questions combining multiple skills.

**Example types:**
- "X0 IsA X1, and X1 is NOT_CapableOf X2. Can X0 do X2?" (inheritance + negation)
- "X0 AtLocation X1, X1 AtLocation X2. X3 also AtLocation X2. Are X0 and X3 in the same general location?" (transitivity + comparison)
- "X0 Causes X1 (weight 0.9), X0 Causes X2 (weight 0.2). What is X0 most likely to cause?" (chain + weight comparison)
- "X0 HasPrerequisite X1, X1 HasPrerequisite X2, but X0 is NOT_CapableOf X2. Can X0 satisfy its prerequisites?" (chain + negation + contradiction detection)

These should be genuinely challenging — the kind of question that requires actually parsing and reasoning through multiple SML elements.

---

## OUTPUT FORMAT FOR EACH EXAMPLE

```json
{
  "question": "The full question text INCLUDING the <sml> block prepended",
  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "answer": "A",
  "category": "simple_lookup",
  "difficulty": "easy",
  "reasoning_type": "single_relation",
  "num_entities": 4,
  "num_relations": 1,
  "num_hops": 1
}
```

---

## LM-EVALUATION-HARNESS TASK CONFIG

Create a YAML config file compatible with lm-eval v0.4+. The task should:

1. Load the JSONL dataset
2. Format each question as a multiple-choice prompt using the model's chat template
3. Prepend the system message: "You are an AI assistant that uses Structured Markup Language (SML) context to ground your reasoning. Always analyze the provided SML block before answering."
4. Score based on log-likelihood of answer choices (standard multiple_choice approach)
5. Report accuracy overall and broken down by category

The prompt format for each question should be:

```
<sml>
{sml_block}
</sml>

{question_text}
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}
```

**Important:** The evaluation should work in log-likelihood mode (comparing probabilities of answer tokens), NOT generative mode. This is how standard benchmarks like ARC work — the model doesn't generate a full response, it just assigns probabilities to each answer choice.

---

## RUNNER SCRIPT

Create `run_sml_eval.py` that:

1. Takes model path as argument (or defaults)
2. Runs lm-eval with the custom task against the specified model
3. Optionally runs against vanilla Qwen3-4B for comparison
4. Prints a comparison table with scores by category

Usage:
```bash
python run_sml_eval.py --model /path/to/finetuned/model
python run_sml_eval.py --model /path/to/finetuned/model --baseline Qwen/Qwen3-4B
```

---

## CRITICAL REQUIREMENTS

1. **ALL entity tokens MUST be opaque.** No semantic words. Use X0, X1, X2... or ent_001, ent_002... consistently. If a single entity leaks semantic meaning (like using "dog_001" instead of "X3"), that question is worthless.

2. **Every question MUST be answerable purely from the SML block.** If you need outside knowledge to answer, the question is wrong. A human who knows SML syntax but nothing about the world should be able to get 100%.

3. **Distractors must be plausible.** Wrong answers should reference real entities from the SML block, not obviously fake ones. The model needs to actually parse the relations to distinguish correct from incorrect.

4. **Vary the SML block complexity.** Some questions should have 2-3 entities and 1 relation (easy). Others should have 5-7 entities and 4-6 relations (hard). Don't make everything the same size.

5. **Vary confidence weights.** Don't always use 0.9. Mix in 0.3, 0.5, 0.7, 1.0. Some questions specifically test weight interpretation.

6. **Include 5-10 "no information" questions** where the correct answer is "Insufficient information" or "None of the above" — the SML block doesn't contain the information needed. This tests whether the model can recognize absence of evidence.

7. **The dataset must be deterministic and reproducible.** No randomization at eval time. The JSONL is static.

8. **Ensure the question text never reveals the answer.** Don't say "X0 is a type of X1 according to the SML" in the question text if the question asks about X0's type. The question should require the model to read the SML to find the relationship.

---

## FILE STRUCTURE

```
sml_opaque_eval/
├── sml_opaque_reasoning.yaml     # lm-eval task config
├── sml_opaque_reasoning.jsonl    # the 100 questions
├── run_sml_eval.py               # convenience runner
└── README.md                     # brief description
```

---

## TESTING

After building, verify:
1. `lm_eval --tasks sml_opaque_reasoning --model hf --model_args pretrained=Qwen/Qwen3-4B,dtype=float16 --device cuda:0 --limit 5` runs without errors
2. Random chance baseline on 4-choice questions = 25%. A model with no SML understanding should score ~25%.
3. A human reading the SML blocks should be able to answer 100% correctly.

---

## EXAMPLE QUESTIONS (one per category for reference)

### Category 1 — Simple Lookup
```json
{
  "question": "<sml>\nE(0|0|0|0|X0|0|0|0.9)\nE(0|0|0|0|X1|0|0|0.9)\nE(0|0|0|0|X2|0|0|0.9)\nR(AtLocation|0|2|0.8|0|0)\n</sml>\n\nAccording to the SML data, where is X0 located?\nA) X1\nB) X2\nC) X0\nD) Not specified",
  "choices": ["X1", "X2", "X0", "Not specified"],
  "answer": "B",
  "category": "simple_lookup"
}
```

### Category 2 — Negation
```json
{
  "question": "<sml>\nE(0|0|0|0|X0|0|0|0.9)\nE(0|0|0|0|X1|0|0|0.9)\nE(0|0|0|0|X2|0|0|0.9)\nR(CapableOf|0|1|0.9|0|0)\nR(NOT_CapableOf|0|2|0.85|0|0)\n</sml>\n\nBased on the SML data, which statement is true?\nA) X0 can do X1 and X2\nB) X0 can do X1 but not X2\nC) X0 can do X2 but not X1\nD) X0 cannot do either",
  "choices": ["X0 can do X1 and X2", "X0 can do X1 but not X2", "X0 can do X2 but not X1", "X0 cannot do either"],
  "answer": "B",
  "category": "negation"
}
```

### Category 3 — Multi-Hop
```json
{
  "question": "<sml>\nE(0|0|0|0|X0|0|0|0.9)\nE(0|0|0|0|X1|0|0|0.9)\nE(0|0|0|0|X2|0|0|0.9)\nE(0|0|0|0|X3|0|0|0.9)\nR(IsA|0|1|0.9|0|0)\nR(CapableOf|1|2|0.85|0|0)\nR(AtLocation|1|3|0.7|0|0)\n</sml>\n\nIf X0 is a type of X1, and X1 can do X2, what can be inferred about X0?\nA) X0 can do X2\nB) X0 is located at X3\nC) X0 is a type of X2\nD) Nothing can be inferred",
  "choices": ["X0 can do X2", "X0 is located at X3", "X0 is a type of X2", "Nothing can be inferred"],
  "answer": "A",
  "category": "multi_hop"
}
```

### Category 4 — Weight Comparison
```json
{
  "question": "<sml>\nE(0|0|0|0|X0|0|0|0.9)\nE(0|0|0|0|X1|0|0|0.9)\nE(0|0|0|0|X2|0|0|0.9)\nR(Causes|0|1|0.9|0|0)\nR(Causes|0|2|0.15|0|0)\n</sml>\n\nAccording to the SML data, what does X0 most strongly cause?\nA) X1\nB) X2\nC) Both equally\nD) X0 does not cause anything",
  "choices": ["X1", "X2", "Both equally", "X0 does not cause anything"],
  "answer": "A",
  "category": "weight_comparison"
}
```

### Category 5 — Counting
```json
{
  "question": "<sml>\nE(0|0|0|0|X0|0|0|0.9)\nE(0|0|0|0|X1|0|0|0.9)\nE(0|0|0|0|X2|0|0|0.9)\nE(0|0|0|0|X3|0|0|0.9)\nR(IsA|0|1|0.8|0|0)\nR(HasA|0|2|0.7|0|0)\nR(CapableOf|1|3|0.6|0|0)\n</sml>\n\nHow many entities in the SML block have at least one outgoing relation?\nA) 1\nB) 2\nC) 3\nD) 4",
  "choices": ["1", "2", "3", "4"],
  "answer": "B",
  "category": "counting"
}
```

### Category 6 — Composite
```json
{
  "question": "<sml>\nE(0|0|0|0|X0|0|0|0.9)\nE(0|0|0|0|X1|0|0|0.9)\nE(0|0|0|0|X2|0|0|0.9)\nE(0|0|0|0|X3|0|0|0.9)\nR(IsA|0|1|0.9|0|0)\nR(CapableOf|1|2|0.85|0|0)\nR(NOT_CapableOf|0|2|0.95|0|0)\n</sml>\n\nX0 is a type of X1, and X1 can do X2. However, X0 specifically cannot do X2. What does this tell us?\nA) X0 is an exception — it's a type of X1 that lacks the X2 capability\nB) The SML data is contradictory and no conclusion can be drawn\nC) X0 can do X2 because it inherits from X1\nD) X1 also cannot do X2",
  "choices": [
    "X0 is an exception that lacks the X2 capability",
    "The data is contradictory, no conclusion possible",
    "X0 can do X2 via inheritance",
    "X1 also cannot do X2"
  ],
  "answer": "A",
  "category": "composite"
}
```

---

## WHAT SUCCESS LOOKS LIKE

- **Random baseline:** ~25% (4-choice random)
- **Vanilla Qwen3-4B (no SML training):** ~25-35% (might get some from pattern matching but mostly guessing)
- **SML fine-tuned model:** Significantly above 50% proves SML training provides real reasoning augmentation
- **SML fine-tuned model > 70%:** Strong evidence the approach works

If fine-tuned and vanilla score the same (~25%), the model hasn't learned to reason from SML structure and the training approach needs fundamental changes.
