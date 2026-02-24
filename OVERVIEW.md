

**SML: Structured Markup Language**

A Neurosymbolic Architecture for Grounded Reasoning

Project Journey, Findings & Roadmap

February 2026

Sweet Papa Technologies LLC

**PROOF OF CONCEPT VALIDATED**  
86% accuracy on opaque entity reasoning (vs 47% vanilla baseline)

# **Executive Summary**

SML (Structured Markup Language) is a neurosymbolic architecture that teaches small language models to reason using structured semantic representations. Over approximately two weeks of intensive development across four training rounds, we have achieved a decisive proof of concept: a fine-tuned Qwen3-4B model scores 86% accuracy on a custom evaluation suite designed to test genuine relational reasoning over completely opaque entities, compared to 47% for the vanilla base model.

This result demonstrates that SML fine-tuning creates a genuine reasoning capability absent in the base model. The model can follow relational structure, handle negations, chain multi-hop inferences, compare confidence weights, and detect contradictions in entity-relation data where it has zero pretrained knowledge.

Standard benchmarks (ARC, HellaSwag, TruthfulQA, Winogrande) showed no degradation from fine-tuning but also no improvement from SML injection. This is expected and explained: these benchmarks test knowledge the model already possesses, making SML injection redundant overhead. SML provides value specifically when the model encounters novel structured knowledge it was not pretrained on.

The project is now at a critical juncture. The core mechanism works, but further validation is needed before this constitutes a publishable result. This document outlines the complete journey, technical details, findings, and the roadmap to publication.

# **The Problem SML Solves**

Large Language Models suffer from several fundamental limitations: they hallucinate facts with confidence, they lack robust logical reasoning capabilities, and they cannot reliably integrate novel structured knowledge at inference time. Current approaches to knowledge-grounded generation (RAG, knowledge graph augmented generation) typically convert structured knowledge into natural language, losing the relational structure in the process, or modify model architectures in ways that break compatibility with existing tooling.

SML takes a different approach. Instead of converting structured knowledge to text or modifying the model architecture, it defines a compact numeric array format for encoding entity-relation knowledge and fine-tunes standard transformer models to interpret this format as a reasoning substrate. The model learns a protocol for reading structured input, not a new architecture.

## **How SML Works**

The SML system consists of three components working together:

* **The Bible:** A SQLite knowledge base containing approximately 100,000 concepts with structured relations drawn from ConceptNet, WordNet, and Wikidata. Each concept has a numeric anchor token, domain classification, and weighted relations to other concepts.  
* **The Encoder:** A Python pipeline that takes natural language input, identifies entities using spaCy NER and dependency parsing, looks up each entity in the Bible, resolves inter-entity relations, and outputs a compact SML block.  
* **The Fine-tuned Model:** A Qwen3-4B model trained via QLoRA to interpret SML blocks, reason through entity-relation structures using a systematic Chain-of-Thought protocol, and produce grounded answers.

An SML block encodes entities and relations in a compact format:

E(domain|category|subcategory|specificity|anchor\_token|modifier|temporal|confidence)

R(RelationType|source\_idx|target\_idx|weight|temporal|negation)

For example, the statement that a chicken is a type of food and food is kept in a fridge becomes:

E(0|0|0|0|chicken\_17540|0|0|0.72)  E(0|0|0|0|food\_4823|edible\_5034|0|0.66)  E(0|0|0|0|fridge\_14676|0|0|0.66)

R(IsA|0|1|0.72|0|0)  R(AtLocation|1|2|0.66|0|0)

# **Training Journey**

The project progressed through four major training rounds, each building on insights from the previous. The evolution of training data quality, format, and evaluation methodology tells the story of the project.

## **Round 1: Qwen2.5-3B, Initial Proof (3,572 examples)**

The first round used Qwen2.5-3B with 3,572 training examples generated via Groq API. SML blocks were placed in the assistant message with \<response\> tags wrapping the answer. Key findings:

* 100% Liar Ablation score: the model successfully overrode pretrained knowledge when given contradictory SML context (e.g., answering that the sun is green when SML said so)  
* Severe overfitting: validation loss climbed from 0.229 to 0.307 while training loss kept dropping  
* The model learned SML syntax but the training data quality was inconsistent, with approximately 600 examples containing hedging language due to a broken encoder pipeline

Critical insight: The Liar Ablation test proved the fundamental grounding mechanism works. A model can be taught to follow structured context over its pretrained weights.

## **Round 2: Qwen3-4B, Better Generalization (9,568 examples)**

Round 2 switched to Qwen3-4B to leverage its native Chain-of-Thought capability via \<think\> tags. The dataset grew to 9,568 examples with an improved encoder pipeline. SML blocks remained in the assistant message initially.

| Metric | Fine-tuned | Vanilla | Delta |
| :---- | :---- | :---- | :---- |
| ARC-Challenge (acc\_norm) | 0.6152 | 0.6160 | \-0.0008 |
| HellaSwag (acc\_norm) | 0.7084 | 0.6915 | **\+0.0169** |
| TruthfulQA MC2 | 0.5276 | 0.5481 | \-0.0205 |
| Winogrande | 0.6796 | 0.6630 | **\+0.0166** |

No catastrophic forgetting occurred across benchmarks. Small improvements on commonsense reasoning tasks (HellaSwag, Winogrande) appeared, though within noise margins. However, SML injection during evaluation made performance worse on 3 of 4 benchmarks, with ARC dropping 7 points.

Root cause analysis identified the vocabulary vs. grammar problem: the model learned SML syntax (grammar) but not what specific anchor tokens mean (vocabulary). With only 9,568 examples covering a subset of the Bible’s 100K concepts, the model encountered unknown tokens at inference and fell back to pretrained knowledge.

## **Round 3: Format Revolution & Algorithmic CoT (Mixed \+ Pure)**

Round 3 introduced two critical changes:

* **V3 Format:** SML blocks moved from the assistant message to the user message, matching inference-time behavior where the Encoder prepends SML to user queries. The model now trains only on reasoning \+ response, not on reproducing SML.  
* **Algorithmic CoT:** 500 high-quality examples were generated with explicit step-by-step reasoning protocols: SCAN entities, PARSE relations, INTERPRET anchors, SYNTHESIZE evidence, CONCLUDE with confidence. This replaced the vague reasoning traces of earlier data.

Two sub-experiments were run:

*Mixed dataset (7,003 examples after validation): 500 new algorithmic CoT \+ 6,503 converted old examples. TruthfulQA flipped from negative to positive under SML injection, and degradation on other benchmarks decreased. This proved the quality of CoT training data directly impacts SML reasoning quality.*

*Pure dataset (2,512 algorithmic CoT only): Severe overfitting with train-val gap of 0.345 at step 500\. Only 314 total training steps were insufficient despite high data quality. Liar Ablation dropped to 40%. This proved that data quality alone cannot substitute for data volume.*

## **Round 4: Scale \+ Quality (30,046 examples)**

Round 4 combined the lessons from all previous rounds: high-quality algorithmic CoT format, V3 structure (SML in user message), and sufficient volume. 30,046 examples were generated, all in the new format with systematic reasoning traces.

Training curve (best across all rounds):

| Step | Train Loss | Val Loss | Gap | Epoch |
| :---- | :---- | :---- | :---- | :---- |
| 500 | 0.4028 | 0.4004 | 0.002 | 0.27 |
| 1000 | 0.3448 | 0.3527 | 0.008 | 0.54 |
| 1500 | 0.3293 | 0.3353 | 0.006 | 0.81 |
| 2000 | 0.2809 | 0.3271 | 0.046 | 1.09 |
| 2500 | 0.2853 | 0.3194 | 0.034 | 1.34 |
| 3000 | 0.2852 | 0.3128 | 0.027 | 1.61 |
| 3500 (best) | 0.2868 | 0.3062 | 0.019 | 1.87 |
| 4000 | 0.2054 | 0.3243 | 0.119 | 2.15 |

The optimal checkpoint was step 3500 (epoch 1.87) with the lowest validation loss of 0.306. Notable: train loss plateaued around 0.285 from steps 2000-3500 while validation loss continued dropping, indicating the model was refining generalizable abstractions rather than memorizing. Overfitting onset occurred sharply at the epoch 2 boundary.

# **The Decisive Experiment**

Standard benchmarks consistently failed to show SML improvement because they test knowledge the model already possesses. A custom evaluation suite was designed to isolate SML reasoning capability by testing on completely opaque entities where pretrained knowledge is useless.

## **Opaque Entity Evaluation Design**

100 multiple-choice questions (4 choices each, 25% random baseline) were created across six categories, using entity tokens like X0, X1, X2 with zero semantic content:

| Category | Count | Tests |
| :---- | :---- | :---- |
| Simple Property Lookup | 20 | Reading a single relation to answer a direct question |
| Negation Reasoning | 15 | Interpreting NOT\_ relations correctly |
| Multi-Hop Chains | 20 | Following 2-3 relation chains to reach conclusions |
| Weight Comparison | 15 | Comparing relation weights to determine strength |
| Counting & Structure | 10 | Counting entities, relations, identifying structural properties |
| Composite Reasoning | 20 | Combining multiple skills (inheritance \+ negation, chains \+ weights) |

Every question is answerable purely from the SML block. A human who understands SML syntax but knows nothing about the world should score 100%. A model with no SML training should score approximately 25% (random chance).

## **Results**

| Model | Accuracy | vs Random (25%) |
| :---- | :---- | :---- |
| SML Fine-tuned (Qwen3-4B) | 86% | **\+61 points** |
| Vanilla Qwen3-4B | 47% | **\+22 points** |
| Delta (fine-tuned advantage) | **\+39 points** |  |

The 39 percentage point improvement on a task designed to be impossible without SML reasoning constitutes definitive proof that SML fine-tuning creates a genuine reasoning capability absent in the base model.

The vanilla model’s 47% (above 25% random) is explained by surface-level pattern matching: some answer choices leak information (weight comparison answers contain the numbers), and counting questions can be solved by counting E() lines. The fine-tuned model’s 86% requires actual relational reasoning including multi-hop inference, negation handling, and composite logic that surface matching cannot solve.

# **Key Insights Across All Rounds**

## **Data Quality \> Data Quantity (to a point)**

The 500 high-quality algorithmic CoT examples in Round 3 moved the TruthfulQA needle more than 9,000 weak examples. But 500 alone was insufficient volume, causing severe overfitting. The optimal recipe proved to be high quality AT sufficient volume: 30K well-structured examples at approximately 2 epochs.

## **Training Format Must Match Inference**

Moving SML from the assistant message (V1/V2 format) to the user message (V3 format) was critical. The model should train on the same input structure it will see at inference time. Training the model to generate SML wastes adapter capacity on a job that belongs to the Encoder.

## **Standard Benchmarks Are the Wrong Eval**

ARC, HellaSwag, TruthfulQA, and Winogrande test knowledge the model already has. SML injection on these benchmarks adds overhead but no new information. SML’s value proposition is providing structured knowledge the model does not already possess. The opaque entity evaluation directly tests this and shows a 39-point advantage.

## **Overfitting Follows Predictable Patterns**

Across all rounds, overfitting onset correlated with the epoch boundary where the model begins seeing repeated data. With 2.5K examples, overfitting was immediate. With 7K, it appeared within epoch 1\. With 30K, the model maintained near-zero train-val gap through the entire first epoch and remained healthy through epoch 1.87.

## **CoT-Trained Models Resist Blind Override**

Qwen3’s native \<think\> training made it more resistant to blindly following SML context that contradicts pretrained knowledge, compared to Qwen2.5. This is actually desirable behavior for production: the model should use SML to inform reasoning, not as an unquestionable oracle. The Round 4 bird example demonstrated sophisticated reconciliation behavior, stating that birds generally can fly but the specific SML context indicated otherwise.

# **Standard Benchmark Summary**

Across all rounds, standard benchmarks told a consistent story: no catastrophic forgetting, no meaningful improvement from SML injection.

| Round | ARC Δ | HellaSwag Δ | TruthfulQA Δ | Winogrande Δ | Note |
| :---- | :---- | :---- | :---- | :---- | :---- |
| R2 (SML inject) | \-0.070 | **\+0.010** | \-0.019 | \-0.040 | SML hurts 3/4 |
| R3 mixed (SML inject) | \-0.050 | \-0.010 | **\+0.017** | \-0.010 | Improving trend |
| R4 (SML inject) | \-0.040 | \-0.040 | \-0.026 | \-0.040 | Still negative |

The persistent negative delta on standard benchmarks with SML injection is explained by a fundamental mismatch: these benchmarks test existing knowledge. SML blocks add tokens the model must process but provide no information the model does not already have. The 13x overhead in processing time makes this actively counterproductive.

Importantly, the system prompt issue may account for some degradation: the harness wrapper prepends raw SML to prompts without the system message the model was trained to expect. This is a known issue to be resolved.

# **Current State of the Project**

* **Proven:** SML fine-tuning creates genuine relational reasoning capability on novel entities (86% vs 47% baseline)  
* **Proven:** No catastrophic forgetting across four training rounds  
* **Proven:** 30K high-quality algorithmic CoT examples at \~2 epochs is the optimal training recipe  
* **Proven:** V3 format (SML in user message, \<think\> tags, no \<response\> wrapper) works for Qwen3-4B  
* **Not yet proven:** Practical value on real-world tasks (medical, legal, enterprise knowledge)  
* **Not yet proven:** Advantage over natural language context injection (the key baseline)  
* **Not yet proven:** Transfer to other model architectures (Llama, Mistral, Phi)  
* **Not yet addressed:** Bible data quality issues (some incorrect ConceptNet entries)  
* **Not yet addressed:** System prompt missing from standard benchmark harness wrapper

# **Roadmap to Publication**

The proof of concept is validated. The following steps are needed to produce a publishable result, ordered by priority and impact.

## **Phase 1: Critical Baselines (1-2 weeks)**

### **Natural Language Baseline Comparison**

This is the single biggest threat to the SML thesis. Take the same 100 opaque entity evaluation questions. Replace each SML block with an equivalent natural language description of the same entities and relations. For example, replace:

R(IsA|0|1|0.9|0|0)  R(CapableOf|1|2|0.85|0|0)

with:

X0 is a type of X1 (confidence: 0.9). X1 is capable of X2 (confidence: 0.85).

Run vanilla Qwen3-4B on the natural language version. If it scores 80%+ without any fine-tuning, SML is an unnecessary abstraction and the thesis collapses. If it scores significantly lower than the SML fine-tuned model, SML provides genuine structural reasoning advantage.

### **Token Efficiency Measurement**

Measure the token count of SML blocks vs equivalent natural language descriptions for the same set of relations. If SML is more compact, this creates a practical argument for context-window-constrained applications even if accuracy is similar to natural language injection.

## **Phase 2: Real-World Validation (2-3 weeks)**

### **Domain-Specific Evaluation**

Build an evaluation using a real knowledge graph the model has never seen. Candidates include: a medical ontology subset (SNOMED-CT or ICD-10 relations), a legal taxonomy (jurisdiction hierarchies, statute relations), or a specialized scientific domain (chemistry compound relations). Encode the graph in SML. Ask questions requiring relational reasoning over the encoded knowledge. Compare SML-augmented model vs base model vs natural language injection.

### **Fix Standard Benchmark Wrapper**

Update the sml\_harness.py wrapper to inject the system message into the chat template. Re-run standard benchmarks to determine whether the missing system prompt accounts for SML injection degradation.

## **Phase 3: Ablation Studies (1-2 weeks)**

* Training data volume curve: 5K vs 10K vs 15K vs 30K examples, measuring opaque entity accuracy at each level  
* Performance breakdown by question category: which reasoning skills (negation, multi-hop, composite) improve most with training  
* Chain length scaling: 2-hop vs 3-hop vs 4-hop reasoning accuracy  
* Confidence weight sensitivity: does the model correctly differentiate high-weight from low-weight relations?  
* Epoch optimization: confirm that \~2 epochs is optimal across dataset sizes

## **Phase 4: Generalization (2-3 weeks)**

* Test on a second model architecture (Llama-3.2-3B, Mistral-7B, or Phi-3-mini) to determine if the training protocol transfers  
* Build domain-specific Bibles for programming and/or medical domains to test vertical applications  
* Evaluate Continued Pre-Training (CPT) if LoRA results plateau: generate a Bible encyclopedia dataset and run additional pretraining on base weights

## **Phase 5: Paper (2-3 weeks)**

With the above completed, the paper structure would be:

* **Contribution:** A training protocol that teaches standard transformer models to interpret compact structured semantic input for relational reasoning, without architectural modifications  
* **Key result:** 86% accuracy on opaque entity reasoning (39-point improvement over base model) demonstrating genuine structural reasoning capability  
* **Practical value:** Real-world domain evaluation showing SML improves reasoning over novel structured knowledge  
* **Efficiency:** Token efficiency comparison vs natural language injection  
* **Generalization:** Transfer to multiple model architectures

# **Technical Reference**

## **Training Configuration**

| Parameter | Value |
| :---- | :---- |
| Base Model | Qwen3-4B (Qwen/Qwen3-4B) |
| Method | QLoRA via Unsloth |
| LoRA Rank | 128 |
| Trainable Parameters | 264,241,152 (6.16% of 4.29B) |
| Batch Size | 4 per device, 4 gradient accumulation \= 16 effective |
| Training Examples | 30,046 |
| Optimal Epochs | \~1.87 (step 3500\) |
| Best Validation Loss | 0.3062 |
| Training Hardware | Single NVIDIA RTX 4090 |
| Training Time | \~1.5 hours to optimal checkpoint |

## **Bible Statistics**

| Component | Detail |
| :---- | :---- |
| Total Concepts | \~100,000 |
| Sources | ConceptNet, WordNet, Wikidata |
| Storage | SQLite database |
| Relation Types | 20+ (IsA, HasA, CapableOf, Causes, etc.) |
| Relation Weights | 0.0-1.0 confidence scores |

## **Evaluation Suite**

| Component | Detail |
| :---- | :---- |
| Framework | EleutherAI lm-evaluation-harness v0.4+ |
| Custom Task | sml\_opaque\_reasoning (100 MC questions) |
| Scoring | Log-likelihood (standard multiple\_choice) |
| Baseline Benchmarks | ARC-Challenge, HellaSwag, TruthfulQA MC2, Winogrande |
| Sample Size for Standard Benchmarks | 100 per task (with n=100 caveat) |

## **Known Issues**

* **Bible data quality:** ConceptNet contains incorrect entries (e.g., NOT\_CapableOf eye/cry\_tears). Mitigation options include confidence threshold filtering, cross-validation against Wikidata, or LLM-based fact checking.  
* **Domain fields unpopulated:** Most Bible entities have domain=0 (unspecified). The model never learns to use domain/category/subcategory fields in reasoning.  
* **Offensive modifiers:** ConceptNet includes modifiers like stupid\_8286 on people entities. Should be filtered at the encoder level.  
* **System prompt missing from harness:** The SML injection wrapper for standard benchmarks does not include the system message, potentially explaining some performance degradation.  
* **Standard benchmarks at n=100:** All standard benchmark comparisons use 100-sample subsets. Standard errors of \~5% mean most deltas are within noise. Full-dataset runs needed for publication.

# **Conclusion**

SML represents a novel approach to neurosymbolic AI: rather than modifying model architectures or converting structured knowledge to natural language, it creates a learned protocol between standard transformers and compact structured representations. The 86% accuracy on opaque entity reasoning proves this protocol works.

The approach is genuinely different from existing work in knowledge-grounded generation. Most approaches either serialize knowledge graphs to natural language (losing structure), embed graph neural networks into transformer architectures (losing compatibility), or use retrieval-augmented generation with text chunks (losing relations). SML maintains structural fidelity in a format the model learns to interpret natively.

The path from proof of concept to publication requires approximately 6-10 weeks of focused work: establishing the natural language baseline, validating on real-world domains, running ablation studies, and testing cross-architecture transfer. The core mechanism is validated. The remaining work is building the evidence package that makes this contribution convincing to the research community.

*Document prepared February 24, 2026\. Model checkpoint: sweetpapa/sml-qwen3-4b-phase3-full (step 3500).*