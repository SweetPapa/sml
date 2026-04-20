# SML v4 Readiness

How the current codebase maps to the v4 approach (SML-as-priming-RAG, per `approach_reveiw_v4.md`), and what we need to build to actually run the experiment.

---

## 1. Where v3 leaves us

The current repo is a **proof-of-concept fine-tuning study**, not a retrieval system. Inference today is:

```
user query → SMLEncoder (spaCy NER + Bible lookup) → SML block → fine-tuned Qwen3-4B → answer
```

The "knowledge" injected into the prompt is **derived from the entities the user already mentioned**, not retrieved from a corpus indexed for relevance. There is no top-K retrieval, no embedding index, no gating, and no comparison baseline beyond the vanilla model.

### 1.1 Inventory of what exists

| Layer | Module | Files | Status |
|---|---|---|---|
| Knowledge base | Bible (SQLite) | `sml/bible/schema.py:18-145`, `sml/bible/query.py:1-120`, `data/sml_bible.db` (53 MB) | ✅ ~100K concepts, 34 ConceptNet relation types, FTS5 fuzzy search. `vector_blob` column exists but **unused**. |
| Encoder | Heuristic NER → SML | `sml/encoder/encoder.py:246-299`, `sml/encoder/formatter.py:45-61` | ✅ spaCy noun chunks + lemma + FTS5 fallback, sentence-embedding disambiguation (MiniLM, lazy-loaded), 3-phase entity + relation resolution. |
| Format | E()/R() spec | `sml/config.py:18-40`, `sml/encoder/formatter.py` | ✅ EDA width 8, RA width 6, NOT_ negation prefix, `<sml>...</sml>` wrapper. No new vocab tokens added. |
| Training data gen | V3 algorithmic CoT | `scripts/02_generate_data.py`, `scripts/07_generate_v3_data.py`, `sml/training/data_generator_v3.py` | ✅ Groq-orchestrated 3-call generator. SML in user message; `<think>` reasoning trace; SCAN→PARSE→INTERPRET→SYNTHESIZE→CONCLUDE template. |
| Training | QLoRA on Qwen3-4B | `scripts/03_train.py`, `sml/config.py` | ✅ Unsloth, LoRA rank 128, 30,046 examples, optimal at step 3500 (val loss 0.306). |
| Inference | End-to-end pipeline | `sml/inference/pipeline.py:1-80`, `scripts/04_inference.py` | ✅ Encodes query → prepends SML → forces `<think>` → decodes. **Single-shot, no retrieval iteration, no gating.** |
| Evaluation | Opaque-entity MC | `sml_opaque_eval/sml_opaque_reasoning.{jsonl,yaml}`, `sml_opaque_eval/run_*.py` | ✅ 100 MC questions over 6 categories. lm-eval-harness compatible. **Headline result: 86% FT vs 47% vanilla.** |

### 1.2 What the validated result actually proves

Current experiments show: a fine-tuned Qwen3-4B can read the SML format and reason over **opaque tokens (X0, X1, ...) it has never seen**, beating its vanilla self by 39 points. That demonstrates the format-decoding capability, not that SML beats matched baselines on real tasks. The v3 OVERVIEW.md itself flags the open question: *"Not yet proven: Advantage over natural language context injection (the key baseline)."*

That open question is exactly what `approach_reveiw_v4.md` is responding to.

---

## 2. What v4 reframes

`approach_reveiw_v4.md` reframes SML from "a fine-tuned reasoning substrate" to **"a retrieved priming annotation."** Three claims must be tested independently, and failing any one falsifies the thesis:

1. **SML injection beats no injection.** (Trivial floor.)
2. **SML beats matched prose RAG** carrying the same content.
3. **SML beats a bullet list of just the retrieved concept names** — the hardest baseline, where most novel-DSL approaches collapse.

The v4 experimental playbook (lines 53–102 of the review) prescribes a **9-condition baseline matrix** on **PopQA, CommonsenseQA, ARC-Challenge, StrategyQA, MuSiQue-Ans, 2WikiMultihopQA** at N≥1000 each, with paired McNemar tests, bootstrapped CIs, popularity-stratified analysis, relation-shuffle ablations, and adversarial SML probes.

The retrieval architecture v4 implies (synthesizing the GraphRAG/HippoRAG/IRCoT/PASTA/PopQA discussion):

```
user query
  ├─→ query embedding / HyDE expansion
  │     ↓
  │   top-K concept retrieval over Bible (FAISS)
  │     ↓
  │   relation-subgraph expansion
  │     ↓
  │   format-as-SML | format-as-prose | format-as-name-bullets   (condition switch)
  │     ↓
  │   adaptive gating (Mallen popularity threshold)
  │     ↓
  │   end-anchored injection adjacent to the query
  └─→ model (vanilla OR LoRA-adapted) → answer + faithfulness/precision metrics
```

None of the boxed steps above currently exist. The v3 encoder is the wrong shape for any of them — it extracts entities *from* the query, not retrieves entities *for* the query.

---

## 3. Gap analysis: v3 vs v4

| v4 requirement | v3 today | Gap |
|---|---|---|
| Embedding index over Bible concepts | `vector_blob` column exists, unpopulated. MiniLM model is loaded for disambiguation only. | Build: backfill embeddings for all ~100K concepts; persist FAISS (or hnswlib) index. |
| Top-K concept retrieval for a query | None. Encoder only resolves entities the user typed. | Build: `Retriever` module with `retrieve(query, k) → [concept_ids]`. |
| Subgraph expansion (1-2 hops from retrieved seeds) | `query.get_outgoing_relations(concept_id)` exists for 1 hop. | Build: `expand_subgraph(seed_ids, hops, max_relations)`. |
| Multiple injection formats (SML / prose / name-bullets / random) | Only SML formatter exists. | Build: `formatters/{sml,prose,bullets,random_sml,random_prose,adversarial}.py`. All must be **token-count-matched** to the SML version. |
| Adaptive gating | None. Always injects. | Build: simple popularity / model-uncertainty gate. Mallen showed always-on retrieval *hurts* head items. |
| HyDE / Query2Doc query expansion | None. | Build: optional pre-retrieval LLM expansion step. |
| Iterative retrieval (IRCoT-style) | Single-shot inference loop only. | Build: stepwise retrieve-then-reason loop, optional. Skip for v4 pilot. |
| Real benchmarks (PopQA, MuSiQue, 2Wiki, CommonsenseQA, ARC, StrategyQA) | Only the custom 100-question opaque suite. | Build: lm-eval task configs that wrap each benchmark with a condition switch. |
| 9-condition baseline matrix | Only conditions (d) full SML / base and (d) full SML / LoRA are implemented. | Build: harness driver that runs the same items through all 9 conditions with paired seeds. Conditions (b), (c), (f), (g), (h), (i) all need formatters, condition (a) is trivial. |
| Popularity stratification on PopQA | n/a | Build: ingest PopQA's popularity field; stratify results into head / torso / tail. |
| Relation-shuffle ablation | n/a | Build: a formatter variant that keeps E() entries but shuffles R() edges. |
| Token-count-matched prose | n/a | Build: prose generator that targets the same token count as the matched SML block. |
| Knows-vs-doesn't-know split | n/a | Build: pre-probe vanilla model per item; cache per-item correctness; slice results by it. |
| Adversarial SML | n/a | Build: retriever variant that returns plausibly-adjacent-but-wrong concepts. |
| Faithfulness / precision / noise sensitivity | n/a | Wire in **RAGAS**. |
| Failure-mode taxonomy (5 categories) | n/a | Build: hand-labeling tooling + 100-item rubric per condition. |
| Attention analysis | n/a | Build: optional transformer-lens / `output_attentions=True` probe on **Qwen3-4B** — both base and the LoRA-adapted checkpoint. Same architecture as the v3 model, so the probe directly tests whether condition (e) shifts attention onto SML concept/relation tokens vs base, instead of running on a separate model and hand-waving the transfer. |
| Statistical reporting (McNemar paired, bootstrap CIs, mixed-effects) | n/a | Build: `analysis/stats.py` shared utility. |
| Frontier-model ICL ceiling test | n/a | Build: a one-page SML legend + 5–10 in-context examples, run on Sonnet/GPT-4.1. The review explicitly calls this the **first** experiment ("$50 experiment that settles the question," line 47). |

The headline gap: **there is no retrieval system at all.** Everything else flows from that.

---

## 4. Build plan

Ordered by dependency, with the smallest decisive experiments first. This deliberately mirrors the v4 review's own ordering (ICL ceiling → pilot → full matrix → diagnostics → write-up).

### 4.1 Phase 0 — ICL ceiling (1–2 days, ~$50)

The cheapest experiment that can falsify the whole thesis. **Do this before building any retrieval infrastructure.**

- `scripts/v4/00_icl_ceiling.py` — pick 100 PopQA items, hand-author or sample 10 SML examples + a one-page schema legend, run Sonnet 4.6 / GPT-4.1 in 3 conditions: no-injection, prose, SML. If a frontier model can't use SML to improve, no amount of FT on a 4B model will rescue it. Decision gate.

### 4.2 Phase 1 — Retrieval infrastructure (~1 week)

This is the core new module. New package: `sml/retrieval/`.

1. **Embedding backfill — two indices, not one.** `scripts/v4/01_embed_bible.py`. Encode every concept with MiniLM (already a dependency) or BGE-small. Build **two** parallel FAISS indices, because the choice of representation is itself an ablation axis:
   - **Index A — surface-only.** Embed `surface_text` (+ definition if present). Cheap baseline; matches what an off-the-shelf concept retriever does.
   - **Index B — relation-expanded.** Embed `surface_text + verbalized top-K outgoing edges`. Lets the retriever surface concepts the question *implies* but doesn't name. Example: a query about "pressure and temperature in gas expansion" should retrieve `ideal_gas_law`, `isothermal`, `adiabatic` — not just the surface tokens it mentions. Without this, the subgraph expander has to do all the work downstream.

   HyDE (§4.2 step 4) expands the *query* side of the same problem; Index B expands the *document* side. They're independent levers; vary them in the ablation cell. Persist both at `data/bible_index_{surface,expanded}.faiss`.
2. **Retriever.** `sml/retrieval/retriever.py` exposing `Retriever.retrieve(query: str, k: int) -> list[ConceptHit]`. Wraps FAISS over the index, with a BM25 alternative (`rank_bm25`) for the prose-vs-structured ablation.
3. **Subgraph expander.** `sml/retrieval/subgraph.py` — given seed concept IDs, walk Bible relations 1–2 hops, capping by total relation count and per-relation weight threshold. Reuses `query.get_outgoing_relations`.
4. **Optional HyDE expansion.** `sml/retrieval/hyde.py` — call a small LLM to draft a hypothetical answer; embed that for retrieval. Behind a flag for an ablation cell.

Decision: do **not** modify the existing `SMLEncoder`. The v3 encoder (entity-extraction-from-query) becomes a fallback / one of several encoder strategies; the v4 retriever is a separate path.

### 4.3 Phase 2 — Condition formatters (~3 days)

New package: `sml/formatters/`. Each formatter takes the same retrieved-subgraph payload and renders one of the 9 conditions. Critical contract: **all variants must target a matched token count** to the SML version on the same payload, otherwise the comparison is contaminated by the "more context = more thinking" confound (review line 89).

| Condition (review §`baseline matrix`) | Formatter | Notes |
|---|---|---|
| (a) No injection | `formatters/none.py` | Pass-through. |
| (b) Prose RAG | `formatters/prose.py` | Verbalize triples; tune length to match SML token count. |
| (c) Concept-name bullets | `formatters/bullets.py` | Names only, no relations. **The hardest baseline to beat.** |
| (d) Full SML | `formatters/sml.py` | Reuses existing `sml.encoder.formatter`. |
| (e) Full SML + LoRA | (same as d, model swap) | No new formatter; runner-level switch. **Conditional run** — only execute (e) after the pilot confirms (d) is beating (c) and (f). If priming itself isn't happening on the base model, "FT improves priming" is unmeasurable. |
| (f) Random SML | `formatters/random_sml.py` | Sample unrelated concepts; preserve structure. Placebo. |
| (g) Human-written considerations | `formatters/oracle.py` | Hand-author for **50 items**, popularity-stratified to match the main PopQA distribution. Pinned size — anything more is scope creep, anything less is too noisy for a defensible ceiling. |
| (h) Random prose | `formatters/random_prose.py` | Prose form of (f). |
| (i) Post-hoc SML | runner-level | Inject after first answer; tests "priming phase" claim. |

Plus the diagnostic formatters: **relation-shuffle** (review line 88) and **adversarial SML** (line 91).

### 4.4 Phase 3 — Benchmark integration & runner (~1 week)

1. **Benchmark loaders.** `sml_v4_eval/datasets/{popqa,musique,twowiki,commonsenseqa,arc,strategyqa}.py`. PopQA needs the popularity field preserved for stratification. Cap at N=1000 each per the review.
2. **lm-eval task templates.** Patch `doc_to_text` to inject the formatter output (the review explicitly recommends this approach, line 97). One YAML per (benchmark × condition) pair, generated from a small template.
3. **Adaptive gating module.** `sml/retrieval/gate.py` — implement a popularity-threshold gate (Mallen) and a confidence-threshold gate (use vanilla model logprob on the no-context answer as a proxy). Logged but off by default in the matrix; on for the gating-ablation cell.
4. **Runner.** `scripts/v4/02_run_matrix.py`. Iterates the 9 conditions × N items × 3 seeds, writes per-item outcomes to a paired-results table (jsonl: `{item_id, condition, seed, predicted, gold, correct, latency_ms, tokens_in, tokens_out}`).
5. **Pre-probe step.** `scripts/v4/03_preprobe_known.py` — run the vanilla model with no injection once per item, cache correctness, used both as condition (a) and as the knows-vs-doesn't-know slicer.

### 4.5 Phase 4 — Metrics & analysis (~3 days)

1. **RAGAS wiring.** `analysis/ragas_metrics.py` — faithfulness, context precision, context recall, noise sensitivity per condition. Faithfulness is not just a logged metric; it's a falsification check (see hypothesis S2 in §4.6) — low faithfulness + high accuracy is the priming signature, high faithfulness + high accuracy means we built compact RAG.
2. **Stats utility.** `analysis/stats.py` — McNemar paired test on discordance, bootstrap 95% CIs (≥10K resamples), Holm-Bonferroni within the primary hypothesis family, mixed-effects model with item-level random intercepts when pooling across benchmarks.
3. **Slicer.** `analysis/slice.py` — popularity strata (head/torso/tail), known/unknown split, per-benchmark, per-condition.
4. **Failure-mode labeler.** `analysis/failures.py` — sample 100 items per condition into a hand-labeling CSV with the 5-category rubric (over-steering, distraction, conflict, relation-hallucination, no-effect).

### 4.6 Phase 5 — Pre-registration & report (~3 days)

1. **Pre-register before seeing test results** (review line 100). Two **co-primary** hypotheses, not one — the priming thesis specifically lives or dies on these two comparisons; everything else is supporting evidence:

   - **H1 (structure beats names).** On PopQA head-popularity items, condition (d) SML beats condition (c) concept-name bullets, ≥3 accuracy points, McNemar p<0.01, bootstrap 95% CI excluding 0. *Bullets are the structurally hardest baseline; if SML doesn't beat them, the structure adds nothing beyond surface concept priming.*
   - **H2 (relevance beats placebo).** On the same slice, (d) beats (f) random-SML by ≥3 points, same statistical bar. *If real SML doesn't beat random SML with the same structure, the priming hypothesis is falsified outright — "the most informative negative result the study can produce" (review line 101).*

   Apply Holm-Bonferroni within {H1, H2}. **Both must hold** to claim the priming thesis. Winning H2 alone (vs random SML) but losing H1 (vs bullets) means we found a structure artifact, not priming.

2. **Pre-registered secondary hypotheses** — these test the *mechanism*, not just the effect. Register before the run:

   - **S1 (knowledge concentration).** Gain from (d) over (a) is **larger** on items the vanilla model already gets right (knows-set) than on items it misses (unknown-set). Test the interaction with McNemar p<0.05. *The priming hypothesis predicts gains concentrate on known items — the model has the answer, SML just redirects attention to it. If gains are uniform or concentrate on unknown items, the method is silently doing fact injection regardless of how it's framed. Same data as the slicer in §4.5, but pre-committed as a falsification test.*
   - **S2 (faithfulness signature).** Faithfulness for (d) is **lower** than for (b) prose RAG, while accuracy is comparable or higher. *Low faithfulness + high accuracy is the priming signature: the model is using its own parametric knowledge, not copying the context. **High** faithfulness + high accuracy means we reinvented RAG in a shorthand. State this as an explicit interpretation rule in the pre-reg, not just as a metric to log.*

3. **Pilot first.** 200 PopQA items, conditions (a)/(b)/(d)/(f) only, to confirm pipeline and measure paired-discordance rate for power analysis (review line 99). This is the unblock-or-stop checkpoint and the gate for whether condition (e) runs at all.
4. **Then full run.** Add (c), (g), (h), (i) and remaining benchmarks. Run (e) only if the pilot confirmed (d) > (c) and (d) > (f) — see §4.3.
5. **Write-up structure:** ceiling test result → pilot effect sizes → full matrix on H1/H2 → mechanism via S1/S2 → diagnostics (relation-shuffle, adversarial, attention probe) → failure-mode taxonomy.

---

## 5. What we keep, what we rebuild, what we shelve

**Keep as-is**
- The Bible (`data/sml_bible.db`, schema, query layer). This is the highest-leverage v3 asset for v4.
- The SML format spec and the `formatter.py` that emits it. Becomes one of N condition formatters.
- The fine-tuned LoRA checkpoint (Qwen3-4B step 3500). Becomes the "model" axis for condition (e).
- `lm-evaluation-harness` integration approach (custom task with patched `doc_to_text`).

**Rebuild around**
- The encoder pipeline. Today's `SMLEncoder` extracts entities from the query; v4 needs a **retriever** that finds entities relevant to the query. The v3 encoder may survive as a fallback or as a hybrid signal, but it is not the v4 entry point.
- The inference pipeline (`sml/inference/pipeline.py`). Becomes condition-aware and retrieval-driven.
- Evaluation. The opaque-entity 100-question suite stays as a regression check on the format-decoding capability, but it is **no longer the headline metric**. PopQA + matched baselines is.

**Shelve / defer**
- **Round 5 of the same format-SFT recipe.** The v4 review is unambiguous: format-FT returns are exhausted (LIMA, Ghosh et al., the chess-LLM study, the Gemini knowledge-injection plateau at 11%). Don't run more rounds of algorithmic-CoT-on-SML hoping for accuracy gains.
- **However — the STaR-style alternative is not the same recipe and is not shelved.** If any FT happens in the v4 timeframe, do this instead: distill rationale traces *from a frontier model using the SML format*, **filter the traces for task correctness**, and fine-tune on those. The gain comes from correct rationales, not from the notation. This is a different recipe from Round 4 and should not be conflated with it. Probably out of scope for the v4 evaluation pipeline itself, but explicitly named here so it doesn't get re-proposed later as "Round 5 with the same recipe and more data."
- Cross-architecture transfer (Llama / Mistral / Phi) until v4 conditions (a)–(d) confirm the priming effect is real on Qwen.
- Iterative IRCoT-style retrieval. Worth doing, but only after the single-shot matrix runs — it's a confound otherwise.

---

## 6. The decision gates

Three gates, in order. Stop and reassess at any failure:

1. **ICL ceiling (Phase 0).** If Sonnet/GPT-4.1 with a schema legend and 5–10 in-context SML examples can't beat their no-injection baseline on 100 PopQA items, **stop the v4 priming program**. The shorthand isn't a useful reasoning scaffold for a frontier model and no downstream retrieval effort recovers it.

   *Reframing if Phase 0 fails:* the v3 result doesn't disappear, it gets re-told. "A 4B model can be trained, via 30K algorithmic-CoT examples, to interpret a novel relational DSL well enough to score 86% on opaque-entity reasoning — a task a frontier model can't do via in-context learning on the same DSL" is itself an interesting finding. It's a small-LM-trainability result, not a priming result. Worth having that framing ready so a negative Phase 0 doesn't feel like the whole program collapsing — it just means the publishable contribution shifts from "priming via retrieved structured cues" to "fine-tuning teaches DSL competence that ICL cannot."

2. **Pilot (Phase 5 step 3).** 200 PopQA items, conditions a/b/d/f. If (d) doesn't beat (f), H2 is falsified and the priming hypothesis is dead — the review explicitly calls this out as "the most informative negative result the study can produce" (line 101). **Publish either way.** Also: condition (e) does not run unless this gate clears.
3. **Full matrix.** Success requires **both** co-primaries (§4.6): (d) beats (c) on PopQA head items (H1, structure beats names) **and** (d) beats (f) (H2, relevance beats placebo), at the pre-registered effect size and significance bar. The review's own success bar (line 109) implies the same: "beats both prose RAG and concept-name bullets … with retained or improved faithfulness metrics." Anything less and we have a complicated way to do standard RAG.

---

## 7. Estimated scope

Assuming one engineer working full-time:

| Phase | Effort | Cumulative |
|---|---|---|
| 0 — ICL ceiling | 1–2 days | day 2 |
| 1 — Retrieval infrastructure | 5–7 days | day 9 |
| 2 — Condition formatters | 3 days | day 12 |
| 3 — Benchmark integration & runner | 5–7 days | day 19 |
| 4 — Metrics & analysis | 3 days | day 22 |
| 5 — Pre-reg, pilot, full run, write-up | 5–7 days | day 29 |

≈ **3–4 weeks** to a publishable positive-or-negative result, consistent with the review's "2–3 week evaluation playbook" plus a buffer for the retrieval build-out the review assumes already exists.
