# SML Micro-PoC Results → Next Steps for AI Coder

**Date:** February 19, 2026  
**Context:** We ran a Micro-PoC of the SML (Semantic Markup Language) neurosymbolic architecture. A Qwen2.5-3B model was fine-tuned on 312 examples using QLoRA via Unsloth. Results: 70% overall (7/10 tests passed), with the critical "Liar Ablation" test confirming the model DOES follow SML context over its own pre-trained weights. The core thesis is validated. Below are the specific fixes and scaling steps needed for the next iteration.

---

## PRIORITY 1: Fix the SML Bible (Encoder Accuracy)

### Problem
Several "Normal Encoding" test failures were actually ENCODER bugs, not model bugs. The model correctly followed the SML it was given — but the Bible gave it wrong facts. Example: the Bible mapped "sun" → `white_3006` instead of `yellow`. The model dutifully said "white," which the test scored as a failure.

### What to Fix

**1a. Audit all existing concept-to-property mappings in the Bible.**  
For every concept that has color/size/capability properties, verify correctness. Specifically:
- `sun` should have primary color property = `yellow` (not white)
- Review ALL entity→property mappings used in the test suite and training data
- Cross-reference against ConceptNet edges with weight ≥ 2.0 for validation

**1b. Add a Bible validation script: `scripts/validate_bible.py`**  
This script should:
- Load the SQLite Bible
- For each concept that has modifier/property relations, print the top 3 properties by weight
- Flag any concept where the assigned property contradicts common knowledge (use a small curated list of ~50 "ground truth" checks: sun=yellow, sky=blue, grass=green, fire=red, snow=white, etc.)
- Output a report of suspect mappings

**1c. Improve the Encoder's property selection logic.**  
When the Encoder builds an EDA and needs to fill modifier slots, it should:
- Query ALL properties for the concept from the `relations` table
- Sort by `weight` descending
- Select the highest-weight property that matches the query context
- For "What color is X?" queries, prioritize `HasProperty` relations where the target concept is a color

---

## PRIORITY 2: Fix Negation Handling

### Problem
The model failed the dogs/bark test in BOTH normal mode ("Can dogs bark?" → should say yes) AND liar mode ("Can dogs bark?" with negated SML → should say no). The negation boolean flag (`Negation=1` in position 5 of the Relation Array) is too weak a signal to override strong pre-trained priors.

### What to Fix

**2a. Make negation more explicit in the SML format.**  
Current format:
```
R(5|0|1|0.9|4|1)
```
The `1` at position 5 means "negated" but the model doesn't attend to it strongly enough.

Change to use an explicit `NOT_` prefix on the relation type label:
```
R(NOT_CapableOf|0|1|0.9|4|0)
```
Or if keeping numeric relation types, add the word directly:
```
R(5_NOT|0|1|0.9|4|0)
```

The string "NOT" activates the model's pre-trained understanding of negation far more effectively than a boolean flag in position 5.

**2b. Add significantly more negation examples to training data.**  
Current training data likely has very few negation examples. Target: at least 15-20% of all training examples should involve negation. Include:
- Animals that CAN'T do common things: "Penguins cannot fly", "Fish cannot walk", "Snakes cannot hear"
- Objects with negated properties: "Ice is not hot", "The night sky is not bright"
- Counter-intuitive negations for Liar testing: "Dogs cannot bark" (false, but model must follow SML)

**2c. Create a dedicated negation test suite: `data/tests/negation_tests.jsonl`**  
At least 20 test cases specifically targeting negation, covering:
- Simple capability negation (can't fly, can't swim)
- Property negation (not red, not large)
- Location negation (not at the beach, not in water)
- Negation that contradicts strong priors (dogs can't bark, birds can't fly)

---

## PRIORITY 3: Enforce the `<thinking>` Block

### Problem
In inference, the model skipped the `<thinking>` block entirely and went straight from `<sml>` to response. The three-stage COT structure (`<sml>` → `<thinking>` → `<response>`) is critical because the `<thinking>` block forces the model to explicitly reference SML entities before generating the answer — which strengthens grounding.

### What to Fix

**3a. Verify every training example has a non-empty `<thinking>` block.**  
Write a validation script that parses all training data JSONL and checks:
- Every assistant message contains `<sml>`, `<thinking>`, and `<response>` tags
- The `<thinking>` block is at least 20 tokens long
- The `<thinking>` block explicitly references at least one SML anchor token (e.g., mentions `dog_4527` or quotes an array value)
- Reject/regenerate any training example that fails these checks

**3b. Make the `<thinking>` block more structured in training data.**  
Instead of freeform thinking, enforce a pattern like:
```
<thinking>
SML entities identified: [dog:4527 (mammal, confidence 0.95), park:2001 (location, confidence 0.9)]
SML relations: [dog AtLocation park (weight 0.8, present tense)]
Reasoning: Based on the SML grounding, the dog is located at the park. The confidence is high (0.8+).
</thinking>
```
This structured thinking format gives the model a consistent template to follow, making it much more likely to produce the block reliably.

**3c. During inference, add the start of `<thinking>` to the prompt.**  
If the model still occasionally skips `<thinking>`, force it by appending the opening tag to the generation prompt:
```python
prompt = f"{system_msg}\n{user_msg}\n{sml_block}\n<thinking>\n"
```
This guarantees the model starts generating inside the thinking block rather than skipping ahead.

---

## PRIORITY 4: Scale Training Data to 1,000-2,000 Examples

### Problem
312 examples was enough to prove the core thesis but not enough for robust performance. The model hedges ("it's important to note...") and occasionally ignores SML, indicating it needs more training signal.

### What to Do

**4a. Target dataset sizes for next iteration:**
- **Round 2 target: 1,500 examples** (5x current)
- Minimum diversity requirements:
  - At least 100 unique concepts from the Bible
  - At least 20 of the 34 relation types represented
  - At least 15% negation examples (see Priority 2)
  - Topic distribution: 30% factual/science, 25% commonsense, 20% spatial/location, 15% causation, 10% properties/attributes

**4b. Use the Inverted Pipeline for data generation:**
1. Sample 1,500 diverse prompts from OpenHermes 2.5 or SlimOrca
2. Run each prompt through the local Python Encoder → produces ground-truth `<sml>` block
3. Send (prompt + sml_block) to the Teacher Model (Groq: llama-3.3-70b or similar) with this system prompt:

```
You are a neurosymbolic reasoning assistant. You will be given a user question 
and a Semantic Markup Language (SML) context block that contains grounded facts.

You must respond in EXACTLY this format:

<thinking>
SML entities identified: [list the entities from the SML block with their anchor tokens]
SML relations: [list the relations and what they mean]
Reasoning: [explain your reasoning, explicitly referencing the SML data]
</thinking>
<response>
[Your answer to the user's question, grounded in the SML facts]
</response>

CRITICAL RULES:
- Your response MUST be grounded in the SML context provided
- You MUST reference specific SML anchor tokens in your thinking
- If the SML says something that contradicts common knowledge, FOLLOW THE SML
- The <thinking> block must be at least 2-3 sentences
- Never skip the <thinking> block
```

4. Combine into training tuples: `{input: prompt, output: <sml>...<thinking>...<response>...}`
5. Validate every tuple (see Priority 3a checks)

**4c. Ensure the Encoder can handle the 1,500 prompts.**  
Before sending to Groq, run the Encoder on all 1,500 prompts and log:
- How many concepts were found in the Bible vs. unknown
- How many relations were successfully encoded
- Average number of entities per SML block
- Average number of relations per SML block
- Any crashes or malformed output

Target: >80% of prompts should produce at least 1 entity and 1 relation. Prompts that produce empty SML blocks should be replaced with prompts the Encoder can handle.

---

## PRIORITY 5: Test Anchor Token Format

### Problem
We haven't validated which anchor token format the model learns best. The current format (`con_dog_4527` or similar) may be suboptimal.

### What to Do

**5a. In the next training run, test two formats side-by-side.**

Format A (current):
```
E(1|12|45|0|con_dog_4527|con_brown_102|0|0.98)
```

Format B (compact):
```
E(1|12|45|0|dog:4527|brown:102|0|0.98)
```

**How to test:** Generate 1,500 training examples. Split into two 750-example sets, identical except for anchor format. Fine-tune two separate LoRA adapters. Run the same evaluation suite on both. Compare:
- Liar Ablation pass rate
- Normal encoding accuracy
- Whether `<thinking>` blocks reference anchor tokens correctly
- Token count per SML block (Format B should be ~20-30% fewer tokens)

**5b. Also test whether the numeric ID is even necessary.**  
Create a third variant (Format C) with NO numeric ID:
```
E(1|12|45|0|dog|brown|0|0.98)
```
If this performs equally well, the numeric IDs add no value and should be dropped entirely. This would simplify the Bible and Encoder significantly.

---

## PRIORITY 6: Improved Evaluation Suite

### Problem
The current eval has only 10 tests, which is too few to draw confident conclusions.

### What to Do

**6a. Expand the test suite to at least 50 tests across 5 categories:**

| Category | Count | What It Tests |
|----------|-------|---------------|
| Normal Encoding | 15 | Model answers correctly when SML is correct |
| Liar Ablation | 15 | Model follows intentionally false SML |
| Negation | 10 | Model handles negated relations correctly |
| Unknown Concepts | 5 | Graceful degradation on out-of-Bible concepts |
| Multi-Entity | 5 | Scenes with 3+ entities and multiple relations |

**6b. Fix the evaluation scoring logic.**  
Current scoring checks for exact keyword match (`expected 'yellow' in response`). This is too brittle. A response like "The sun appears yellowish-white" should count as a partial pass. Consider:
- Primary check: Does the response mention the SML-grounded answer?
- Secondary check: Does the response contradict the SML?  
- Score: PASS (follows SML), PARTIAL (mentions SML answer but hedges), FAIL (contradicts SML)

**6c. Add metrics tracking across runs.**  
Create a `data/eval_results/` directory. After each eval run, save:
```json
{
  "run_id": "2026-02-19_v2",
  "model": "qwen2.5-3b-instruct",
  "training_examples": 1500,
  "epochs": 5,
  "lora_rank": 64,
  "results": {
    "normal_encoding": {"pass": 12, "partial": 2, "fail": 1},
    "liar_ablation": {"pass": 13, "partial": 1, "fail": 1},
    "negation": {"pass": 7, "partial": 1, "fail": 2},
    "unknown": {"pass": 5, "partial": 0, "fail": 0},
    "multi_entity": {"pass": 3, "partial": 1, "fail": 1}
  }
}
```
This lets us track improvement across iterations.

---

## PRIORITY 7: Training Hyperparameter Adjustments

### Adjustments for Next Run

Based on the Micro-PoC results:

```python
# Increase LoRA rank from 64 to 128
# Rationale: Learning a new "language" (SML) requires more adapter capacity
r = 128
lora_alpha = 256  # Keep 2:1 ratio with rank

# Reduce dropout to 0 (Unsloth warning showed dropout=0.05 was preventing fast patching)
lora_dropout = 0.0

# Keep 5 epochs for 1,500 examples
# If loss plateaus before epoch 5, add early stopping
num_train_epochs = 5

# Consider slightly lower learning rate for stability with more data
learning_rate = 1.5e-4  # Down from 2e-4

# Increase max_seq_length if SML blocks are being truncated
max_seq_length = 4096  # Up from whatever current value is
```

### Loss Masking Experiment
Run TWO training configurations:
- **Config A:** Train on full output (sml + thinking + response) — current approach
- **Config B:** Mask loss on the `<sml>` block, only train on `<thinking>` + `<response>`

Compare Liar Ablation results. Config B should theoretically perform better because the model learns to READ SML (which is its inference-time job) rather than WRITE SML (which the Encoder handles).

---

## Execution Order

```
Step 1: Fix Bible mappings (Priority 1)           → 1-2 hours
Step 2: Fix negation format (Priority 2a)          → 1 hour  
Step 3: Validate/fix training data format (3a, 3b) → 2 hours
Step 4: Scale Encoder + generate 1,500 examples     → 4-8 hours (Groq API time)
Step 5: Validate all training examples (3a checks)  → 1 hour
Step 6: Expand eval suite to 50 tests (Priority 6)  → 2 hours
Step 7: Train Round 2 (1,500 examples, 5 epochs)    → ~2-3 hours on RTX 4090
Step 8: Run full eval suite                          → 30 minutes
Step 9: Run anchor token format test (Priority 5)    → Additional training runs
```

**Expected outcome after Round 2:** 85%+ overall score (up from 70%), with Liar Ablation at 90%+ and proper `<thinking>` blocks in every response.

---

## Success Criteria for Round 2

| Metric | Round 1 (Current) | Round 2 Target | Notes |
|--------|-------------------|----------------|-------|
| Normal Encoding | 60% (3/5) | 85%+ | Mostly Bible/Encoder fixes |
| Liar Ablation | 67% (2/3) | 90%+ | More training data + negation fixes |
| Negation-specific | Not tested | 80%+ | New test category |
| Unknown Concepts | 100% (2/2) | 100% | Should maintain |
| `<thinking>` block present | ~50% | 95%+ | Format enforcement in training data |
| Overall | 70% (7/10) | 85%+ (43/50) | Across expanded 50-test suite |

---

*If Round 2 hits 85%+, we proceed to full Phase 2: scale to 25,000 examples, import full ConceptNet/WordNet Bible, and target production-grade performance.*
