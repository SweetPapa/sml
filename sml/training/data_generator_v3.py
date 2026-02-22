"""V3 Training data generator — 3-call LLM orchestrator with algorithmic CoT."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sml.training.cluster_selector import ConceptCluster

from sml.config import (
    GROQ_PARALLEL,
    RELATION_TYPES,
    V3_GROQ_CONFIG,
    V3_MANIFEST_PATH,
    V3_SYSTEM_PROMPT,
    V3_TRAINING_DATA_PATH,
)
from sml.encoder.formatter import parse_sml_block
from sml.training.data_generator import (
    _ENTITY_ID_RE,
    _PUNT_PHRASES,
    _RELATION_NAMES,
    _clean_response,
)

logger = logging.getLogger(__name__)


# ── LLM Prompt Templates ────────────────────────────────────────────────────

QUESTION_GEN_SYSTEM = (
    "You are a question writer for an AI training dataset. Given a set of "
    "facts about concepts and their relationships, write a single natural "
    "question that can be answered using those facts. The question should "
    "sound like something a curious person would ask — not a quiz or test.\n\n"
    "RULES:\n"
    "- Write exactly ONE question\n"
    "- The question must end with a question mark\n"
    "- Do NOT mention technical terms like 'entity', 'relation', 'anchor', "
    "'SML', 'knowledge base', or any internal identifiers\n"
    "- The question should be answerable from the provided facts\n"
    "- Keep it between 10 and 300 characters\n"
    "- Vary question types: yes/no, factual, causal, comparison, what-if"
)

QUESTION_GEN_USER_TEMPLATES = {
    "A": (
        "Facts about {seed_name}:\n{facts}\n\n"
        "Write a factual question about {seed_name} that can be answered "
        "using these facts. Vary the style — try yes/no, 'what', 'how', "
        "'why', or 'describe' questions."
    ),
    "B": (
        "Imagine a novel entity called '{seed_name}'. Here are some things "
        "we know about it:\n{facts}\n\n"
        "Write a question about '{seed_name}' as if it were a real thing "
        "someone just discovered. The question should test whether someone "
        "can reason about this new entity from the given facts."
    ),
    "C": (
        "Here is a chain of related concepts:\n{facts}\n\n"
        "Write a question that requires connecting at least two of these "
        "relationships to answer. The question should require multi-step "
        "reasoning — not just looking up a single fact."
    ),
    "D": (
        "Facts about {seed_name} (note: some facts state what it CANNOT do "
        "or does NOT have):\n{facts}\n\n"
        "Write a question that tests understanding of what {seed_name} "
        "cannot do, does not have, or is NOT. The answer should involve "
        "negation or contrast."
    ),
}

# Question type rotation hints appended to user prompt
_QUESTION_TYPES = [
    "Write a yes/no question.",
    "Write a 'what' question.",
    "Write a 'why' or 'how' question.",
    "Write a comparison question.",
    "Write a 'can X do Y?' question.",
]

COT_GEN_SYSTEM = (
    "You are an expert at interpreting Structured Markup Language (SML) — "
    "a compact notation for encoding knowledge.\n\n"
    "SML FORMAT REFERENCE:\n"
    "- Entity: E(domain|category|subcategory|specificity|anchor|mod1|mod2|confidence)\n"
    "  domain: 1=physical, 2=abstract, 3=digital, 4=event, 5=fiction\n"
    "  anchor: unique identifier like 'dog_2451' or 'love_8832'\n"
    "  confidence: 0.0-1.0 reliability score\n\n"
    "- Relation: R(type|source_idx|target_idx|weight|temporal|negation)\n"
    "  type: relation name (IsA, PartOf, HasA, HasProperty, CapableOf, "
    "AtLocation, Causes, HasPrerequisite, UsedFor, CreatedBy, MadeOf, "
    "Desires, CausesDesire, RelatedTo, SimilarTo, Antonym, etc.)\n"
    "  source_idx/target_idx: 0-based index into entity list\n"
    "  weight: 0.0-1.0 confidence in the relationship\n"
    "  negation: 1 means NOT (e.g., NOT_CapableOf = cannot do)\n\n"
    "YOUR TASK: Given an SML block and a question, write a chain-of-thought "
    "reasoning trace that demonstrates step-by-step SML interpretation.\n\n"
    "PROTOCOL (follow this structure but vary the wording):\n"
    "1. SCAN: Note what entities and relations are present\n"
    "2. PARSE ENTITIES: Identify key anchors and their domains\n"
    "3. INTERPRET: Explain what the anchors represent in plain terms\n"
    "4. PARSE RELATIONS: Walk through relevant relations, noting types "
    "and weights\n"
    "5. SYNTHESIZE: Connect the SML data to answer the question\n"
    "6. CONCLUDE: State your answer with confidence based on weights\n\n"
    "RULES:\n"
    "- Reference specific anchor tokens (e.g., dog_2451) and relation "
    "types (e.g., CapableOf)\n"
    "- Use weights to express confidence ('strong signal at 0.92' vs "
    "'weak association at 0.34')\n"
    "- If negation=1, explicitly note the NOT meaning\n"
    "- Vary your style — don't always use the same phrases\n"
    "- Write at least 50 words\n"
    "- Do NOT use <think> or <response> tags — just write the reasoning"
)

COT_GEN_USER = (
    "SML Block:\n{sml_block}\n\n"
    "Question: {question}\n\n"
    "Ground truth context (for your reference only — do NOT quote this "
    "directly):\n{ground_truth}\n\n"
    "Write a chain-of-thought reasoning trace that interprets the SML "
    "block step by step to answer this question. Reference specific "
    "anchors and relation types from the SML."
)

ANSWER_GEN_SYSTEM = (
    "You are a helpful AI assistant. Write a clear, natural-language "
    "answer to the question based on the reasoning provided.\n\n"
    "RULES:\n"
    "- Write conversationally, like a knowledgeable friend\n"
    "- NEVER mention SML, entity IDs, anchor tokens, confidence scores, "
    "relation types, or any technical markup\n"
    "- Keep the answer between 20 and 1000 characters\n"
    "- Be direct and helpful — don't hedge unnecessarily\n"
    "- Do NOT say 'based on the reasoning' or 'according to the data'"
)

ANSWER_GEN_USER = (
    "Question: {question}\n\n"
    "Reasoning summary: {reasoning_summary}\n\n"
    "Write a clear, natural answer to this question. Do not reference "
    "any technical details or internal identifiers."
)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class V3Example:
    """A single V3 training example in progress."""

    cluster: ConceptCluster
    question: str = ""
    reasoning: str = ""
    answer: str = ""
    category: str = ""
    is_valid: bool = False
    retry_count: int = 0
    error: str = ""


# ── Rate limiter (same pattern as data_generator.py) ─────────────────────────

class _RateLimiter:
    """Adaptive rate limiter that reads Groq response headers."""

    def __init__(self, max_concurrent: int, rpm_target: int, tpm_budget: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.rpm_target = rpm_target
        self.tpm_budget = tpm_budget
        self.min_delay = 60.0 / rpm_target
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()
        self._remaining_tokens = tpm_budget
        self._remaining_requests = 10000

    async def acquire(self):
        await self.semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.min_delay:
                await asyncio.sleep(self.min_delay - elapsed)
            self._last_request_time = time.monotonic()

    def release(self):
        self.semaphore.release()

    def update_from_headers(self, headers: dict):
        try:
            rt = headers.get("x-ratelimit-remaining-tokens")
            if rt is not None:
                self._remaining_tokens = int(rt)
            rr = headers.get("x-ratelimit-remaining-requests")
            if rr is not None:
                self._remaining_requests = int(rr)
        except (ValueError, TypeError):
            pass

        if self._remaining_tokens < 20000:
            self.semaphore = asyncio.Semaphore(min(5, self.max_concurrent))
            self.min_delay = 2.0
            logger.warning(
                "Tokens low (%d remaining) — reducing to 5 concurrent, 2s delay",
                self._remaining_tokens,
            )
        elif self._remaining_tokens < 50000:
            self.semaphore = asyncio.Semaphore(min(10, self.max_concurrent))
            self.min_delay = 1.0
        else:
            self.min_delay = 60.0 / self.rpm_target

        if self._remaining_requests < 100:
            logger.warning(
                "RPD low (%d remaining) — consider pausing",
                self._remaining_requests,
            )


# ── Groq API call ────────────────────────────────────────────────────────────

async def _call_groq(
    client,
    system: str,
    user: str,
    rate_limiter: _RateLimiter,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
) -> Optional[str]:
    """Call Groq API with exponential backoff + jitter on 429/503."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    for attempt in range(max_retries):
        await rate_limiter.acquire()
        try:
            completion = client.chat.completions.create(
                model=V3_GROQ_CONFIG["model"],
                messages=messages,
                max_tokens=V3_GROQ_CONFIG["max_tokens"],
                temperature=V3_GROQ_CONFIG["temperature"],
            )

            raw = getattr(completion, "_raw_response", None) or getattr(
                completion, "response", None
            )
            if raw and hasattr(raw, "headers"):
                rate_limiter.update_from_headers(dict(raw.headers))

            return completion.choices[0].message.content

        except Exception as e:
            err_str = str(e)
            status = getattr(e, "status_code", None) or getattr(e, "status", None)

            if status == 429 or "429" in err_str:
                wait = initial_backoff * (2**attempt) + random.uniform(0, 1)
                logger.info("429 rate-limited, retry %d/%d in %.1fs", attempt + 1, max_retries, wait)
                await asyncio.sleep(wait)
            elif status == 503 or "503" in err_str:
                wait = 5.0 + random.uniform(0, 2)
                logger.info("503 unavailable, retry %d/%d in %.1fs", attempt + 1, max_retries, wait)
                await asyncio.sleep(wait)
            elif "blocked_api_access" in err_str:
                logger.error("Spend limit hit — stopping.")
                return None
            else:
                logger.error("Groq error (attempt %d): %s", attempt + 1, err_str)
                if attempt < max_retries - 1:
                    await asyncio.sleep(initial_backoff * (2**attempt))
                else:
                    return None
        finally:
            rate_limiter.release()

    return None


# ── Cluster metadata to plain-English facts ──────────────────────────────────

def _cluster_to_facts(cluster) -> str:
    """Convert cluster concepts/relations to natural-language facts."""
    lines = []
    concepts = cluster.concepts
    relations = cluster.relations

    concept_by_id = {c["id"]: c for c in concepts}

    for rel in relations:
        src = concept_by_id.get(rel["source_id"], {})
        tgt = concept_by_id.get(rel["target_id"], {})
        rel_type_id = rel["relation_type_id"]
        rel_name = RELATION_TYPES.get(rel_type_id, str(rel_type_id))
        negation = rel.get("negation", 0)
        weight = rel.get("weight", 0.5)

        src_name = src.get("surface_text", "?")
        tgt_name = tgt.get("surface_text", "?")

        if negation:
            lines.append(f"- {src_name} does NOT {rel_name} {tgt_name} (confidence: {weight})")
        else:
            lines.append(f"- {src_name} {rel_name} {tgt_name} (confidence: {weight})")

    if not lines:
        for c in concepts:
            lines.append(f"- {c['surface_text']}: {c.get('definition', 'a concept')}")

    return "\n".join(lines)


def _cluster_to_ground_truth(cluster) -> str:
    """Build ground truth context string for CoT generation."""
    lines = []
    for c in cluster.concepts:
        lines.append(
            f"Entity: {c['surface_text']} (anchor={c['anchor_token']}, "
            f"domain={c.get('domain', 0)})"
        )
    for rel in cluster.relations:
        rel_name = RELATION_TYPES.get(rel["relation_type_id"], str(rel["relation_type_id"]))
        neg = "NOT_" if rel.get("negation", 0) else ""
        lines.append(
            f"Relation: {neg}{rel_name} from entity[{rel['source_id']}] "
            f"to entity[{rel['target_id']}] weight={rel.get('weight', 0.5)}"
        )
    return "\n".join(lines)


# ── Inline validation ────────────────────────────────────────────────────────

def _validate_question(question: str, cluster) -> Optional[str]:
    """Validate generated question. Returns error string or None."""
    if not question:
        return "empty question"
    if len(question) < 10:
        return f"too short ({len(question)} chars)"
    if len(question) > 300:
        return f"too long ({len(question)} chars)"
    if not question.strip().endswith("?"):
        return "does not end with ?"
    # Check for anchor tokens leaking
    if _ENTITY_ID_RE.search(question):
        return "contains anchor token"
    # Check for relation names leaking
    q_lower = question.lower()
    for rn in _RELATION_NAMES:
        if rn.lower() in q_lower:
            return f"contains relation name '{rn}'"
    return None


def _validate_reasoning(reasoning: str, cluster) -> Optional[str]:
    """Validate generated reasoning. Returns error string or None."""
    if not reasoning:
        return "empty reasoning"
    words = reasoning.split()
    if len(words) < 50:
        return f"too short ({len(words)} words)"
    # Must reference at least 1 anchor from SML
    anchors = [c["anchor_token"] for c in cluster.concepts if c.get("anchor_token")]
    if not any(a in reasoning for a in anchors):
        return "does not reference any anchor"
    # Must reference at least 1 relation type
    found_rel = False
    for rn in _RELATION_NAMES:
        if rn in reasoning:
            found_rel = True
            break
    if not found_rel:
        return "does not reference any relation type"
    return None


def _validate_answer(answer: str) -> Optional[str]:
    """Validate generated answer. Returns error string or None."""
    if not answer:
        return "empty answer"
    if len(answer) < 20:
        return f"too short ({len(answer)} chars)"
    if len(answer) > 1000:
        return f"too long ({len(answer)} chars)"
    # No anchor tokens
    if _ENTITY_ID_RE.search(answer):
        return "contains anchor token"
    # No relation names
    a_lower = answer.lower()
    for rn in _RELATION_NAMES:
        if rn.lower() in a_lower and rn.lower() not in ("related to", "similar to", "part of"):
            return f"contains relation name '{rn}'"
    # No SML jargon
    for phrase in ["SML", "anchor token", "confidence score", "relation type",
                   "entity descriptor", "knowledge base encoding"]:
        if phrase.lower() in a_lower:
            return f"contains jargon '{phrase}'"
    # No punt phrases
    if any(p in a_lower for p in _PUNT_PHRASES):
        return "punt response"
    return None


# ── Per-step generation ──────────────────────────────────────────────────────

async def _generate_question(
    example: V3Example,
    client,
    rate_limiter: _RateLimiter,
    max_retries: int = 3,
) -> bool:
    """Generate question for a V3 example. Returns True on success."""
    cluster = example.cluster
    facts = _cluster_to_facts(cluster)
    seed_name = cluster.seed_concept.get("surface_text", "this concept")
    template = QUESTION_GEN_USER_TEMPLATES.get(example.category, QUESTION_GEN_USER_TEMPLATES["A"])
    q_type_hint = _QUESTION_TYPES[hash(seed_name) % len(_QUESTION_TYPES)]

    user_msg = template.format(seed_name=seed_name, facts=facts) + f"\n\n{q_type_hint}"

    for attempt in range(max_retries):
        content = await _call_groq(client, QUESTION_GEN_SYSTEM, user_msg, rate_limiter)
        if not content:
            example.retry_count += 1
            continue

        # Extract just the question (take first line ending with ?)
        question = ""
        for line in content.strip().split("\n"):
            line = line.strip().strip('"').strip("'")
            if line.endswith("?"):
                question = line
                break
        if not question:
            question = content.strip().split("\n")[0].strip().strip('"').strip("'")

        error = _validate_question(question, cluster)
        if error:
            logger.debug("Question validation failed (attempt %d): %s — %s", attempt + 1, error, question[:100])
            example.retry_count += 1
            continue

        example.question = question
        return True

    example.error = "question generation failed after retries"
    return False


async def _generate_reasoning(
    example: V3Example,
    client,
    rate_limiter: _RateLimiter,
    max_retries: int = 3,
) -> bool:
    """Generate CoT reasoning for a V3 example. Returns True on success."""
    cluster = example.cluster
    ground_truth = _cluster_to_ground_truth(cluster)

    user_msg = COT_GEN_USER.format(
        sml_block=cluster.sml_block,
        question=example.question,
        ground_truth=ground_truth,
    )

    for attempt in range(max_retries):
        content = await _call_groq(client, COT_GEN_SYSTEM, user_msg, rate_limiter)
        if not content:
            example.retry_count += 1
            continue

        # Strip any accidental tags
        reasoning = content.strip()
        reasoning = re.sub(r"</?think>", "", reasoning).strip()
        reasoning = re.sub(r"</?response>", "", reasoning).strip()

        error = _validate_reasoning(reasoning, cluster)
        if error:
            logger.debug("Reasoning validation failed (attempt %d): %s", attempt + 1, error)
            example.retry_count += 1
            continue

        example.reasoning = reasoning
        return True

    example.error = "reasoning generation failed after retries"
    return False


async def _generate_answer(
    example: V3Example,
    client,
    rate_limiter: _RateLimiter,
    max_retries: int = 3,
) -> bool:
    """Generate final answer for a V3 example. Returns True on success."""
    # Summarize reasoning (take last 2-3 sentences for the answer prompt)
    reasoning_sentences = example.reasoning.replace("\n", " ").split(". ")
    summary = ". ".join(reasoning_sentences[-3:]).strip()
    if not summary.endswith("."):
        summary += "."

    user_msg = ANSWER_GEN_USER.format(
        question=example.question,
        reasoning_summary=summary,
    )

    for attempt in range(max_retries):
        content = await _call_groq(client, ANSWER_GEN_SYSTEM, user_msg, rate_limiter)
        if not content:
            example.retry_count += 1
            continue

        answer = content.strip()
        answer = re.sub(r"</?response>", "", answer).strip()
        answer = _clean_response(answer)

        error = _validate_answer(answer)
        if error:
            logger.debug("Answer validation failed (attempt %d): %s", attempt + 1, error)
            example.retry_count += 1
            continue

        example.answer = answer
        return True

    example.error = "answer generation failed after retries"
    return False


# ── Per-example orchestrator ─────────────────────────────────────────────────

async def _process_example(
    example: V3Example,
    client,
    rate_limiter: _RateLimiter,
    max_retries_per_step: int = 3,
) -> bool:
    """Run 3 sequential LLM steps for one example. Returns True on success."""
    if not await _generate_question(example, client, rate_limiter, max_retries_per_step):
        return False
    if not await _generate_reasoning(example, client, rate_limiter, max_retries_per_step):
        return False
    if not await _generate_answer(example, client, rate_limiter, max_retries_per_step):
        return False

    example.is_valid = True
    return True


# ── Assembly ─────────────────────────────────────────────────────────────────

def _assemble_v3_messages(example: V3Example) -> dict:
    """Build the final JSONL record from a completed V3Example."""
    cluster = example.cluster
    user_content = f"{cluster.sml_block}\n\n{example.question}"
    assistant_content = f"<think>\n{example.reasoning}\n</think>\n\n{example.answer}"

    return {
        "messages": [
            {"role": "system", "content": V3_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "category": example.category,
            "seed_concept": cluster.seed_concept.get("surface_text", ""),
            "num_entities": len(cluster.concepts),
            "num_relations": len(cluster.relations),
            "retry_count": example.retry_count,
        },
    }


# ── Post-assembly validation ─────────────────────────────────────────────────

def _validate_v3_example(record: dict) -> list[str]:
    """Post-assembly validation. Returns list of errors (empty = valid)."""
    errors = []

    # Structure check
    msgs = record.get("messages", [])
    if len(msgs) != 3:
        errors.append(f"expected 3 messages, got {len(msgs)}")
        return errors

    if msgs[0]["role"] != "system":
        errors.append(f"message 0 should be system, got {msgs[0]['role']}")
    if msgs[1]["role"] != "user":
        errors.append(f"message 1 should be user, got {msgs[1]['role']}")
    if msgs[2]["role"] != "assistant":
        errors.append(f"message 2 should be assistant, got {msgs[2]['role']}")

    user_content = msgs[1]["content"]
    assistant_content = msgs[2]["content"]

    # User message must contain <sml>...</sml> + question
    if "<sml>" not in user_content or "</sml>" not in user_content:
        errors.append("user message missing <sml> block")
    if "?" not in user_content:
        errors.append("user message missing question")

    # Assistant must have <think>...</think> + answer, NO <response> tags
    if "<think>" not in assistant_content or "</think>" not in assistant_content:
        errors.append("assistant missing <think> block")
    if "<response>" in assistant_content:
        errors.append("assistant contains <response> tags (V3 removes these)")

    # SML parseable
    sml_match = re.search(r"<sml>.*?</sml>", user_content, re.DOTALL)
    if sml_match:
        try:
            parsed = parse_sml_block(sml_match.group())
            entities = parsed["entities"]
            relations = parsed["relations"]

            for e in entities:
                if len(e) != 8:
                    errors.append(f"EDA width {len(e)} != 8")
                    break

            for r in relations:
                if len(r) != 6:
                    errors.append(f"RA width {len(r)} != 6")
                    break
                # source/target refs in valid range
                src_idx = r[1]
                tgt_idx = r[2]
                if isinstance(src_idx, int) and (src_idx < 0 or src_idx >= len(entities)):
                    errors.append(f"RA source ref {src_idx} out of range (0-{len(entities)-1})")
                if isinstance(tgt_idx, int) and (tgt_idx < 0 or tgt_idx >= len(entities)):
                    errors.append(f"RA target ref {tgt_idx} out of range (0-{len(entities)-1})")
        except Exception as e:
            errors.append(f"SML parse error: {e}")
    else:
        errors.append("could not extract SML block from user message")

    # Think block word count
    think_match = re.search(r"<think>(.*?)</think>", assistant_content, re.DOTALL)
    if think_match:
        think_words = think_match.group(1).split()
        if len(think_words) < 50:
            errors.append(f"think block too short ({len(think_words)} words)")
    else:
        errors.append("could not extract think block")

    # Answer text (after </think>)
    after_think = assistant_content.split("</think>")[-1].strip() if "</think>" in assistant_content else ""
    if len(after_think) < 20:
        errors.append(f"answer too short ({len(after_think)} chars)")

    # No jargon in answer
    a_lower = after_think.lower()
    if _ENTITY_ID_RE.search(after_think):
        errors.append("answer contains anchor tokens")
    for phrase in ["SML", "anchor token", "confidence score"]:
        if phrase.lower() in a_lower:
            errors.append(f"answer contains jargon '{phrase}'")

    return errors


# ── Main generation pipeline ─────────────────────────────────────────────────

async def _generate_all(
    clusters: list,
    client,
    rate_limiter: _RateLimiter,
    output_path: str,
    max_retries_per_step: int = 3,
) -> dict:
    """Run all examples concurrently and write results incrementally."""
    examples = []
    for cluster in clusters:
        ex = V3Example(cluster=cluster, category=cluster.category)
        examples.append(ex)

    generated = 0
    failed = 0
    validation_errors = 0
    total = len(examples)
    category_counts = {}
    start_time = time.monotonic()

    with open(output_path, "w") as f:
        tasks = [
            asyncio.create_task(
                _process_example(ex, client, rate_limiter, max_retries_per_step)
            )
            for ex in examples
        ]

        # Wait for all tasks with progress tracking
        done_count = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done_count += 1
            if done_count % 25 == 0 or done_count == total:
                elapsed = time.monotonic() - start_time
                print(
                    f"  Progress: {done_count}/{total} "
                    f"(elapsed={elapsed:.0f}s)"
                )

        # Write all valid examples
        for ex in examples:
            if ex.is_valid:
                record = _assemble_v3_messages(ex)
                errors = _validate_v3_example(record)
                if errors:
                    logger.debug("Post-assembly validation failed: %s", errors)
                    validation_errors += 1
                else:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    generated += 1
                    cat = ex.category
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            else:
                failed += 1

    elapsed = time.monotonic() - start_time
    total_retries = sum(ex.retry_count for ex in examples)

    return {
        "total": total,
        "generated": generated,
        "failed": failed,
        "validation_errors": validation_errors,
        "by_category": category_counts,
        "total_retries": total_retries,
        "elapsed_seconds": round(elapsed, 1),
    }


def generate_v3_training_data(
    bible_path: str,
    groq_api_key: str,
    output_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    category_counts: Optional[dict[str, int]] = None,
    seed: int = 42,
    max_retries_per_step: int = 3,
    validate: bool = False,
) -> str:
    """Main entry point for V3 training data generation.

    1. Select concept clusters from Bible (no API needed)
    2. Run 3-call LLM pipeline for each cluster concurrently
    3. Validate and write JSONL + manifest

    Returns path to the output JSONL file.
    """
    from groq import Groq

    from sml.training.cluster_selector import ClusterSelector

    output_path = output_path or str(V3_TRAINING_DATA_PATH)
    manifest_path = manifest_path or str(V3_MANIFEST_PATH)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if category_counts is None:
        from sml.config import V3_CATEGORY_DISTRIBUTION
        category_counts = dict(V3_CATEGORY_DISTRIBUTION)

    total_examples = sum(category_counts.values())

    # Phase 1: Cluster selection
    print(f"\nPhase 1: Selecting {total_examples} concept clusters...")
    cs = ClusterSelector(bible_path, seed=seed)
    clusters = cs.select_clusters(category_counts)
    cs.close()

    actual_by_cat = {}
    for c in clusters:
        actual_by_cat[c.category] = actual_by_cat.get(c.category, 0) + 1
    print(f"  Selected {len(clusters)} clusters: {actual_by_cat}")

    # Phase 2: LLM generation
    max_concurrent = GROQ_PARALLEL["max_concurrent"]
    rpm_target = GROQ_PARALLEL["rpm_target"]
    tpm_budget = GROQ_PARALLEL["tpm_budget"]

    print(f"\nPhase 2: Generating V3 examples ({len(clusters)} clusters, "
          f"~{len(clusters) * 3} API calls)...")
    print(f"  Config: {max_concurrent} concurrent, {rpm_target} RPM, "
          f"model={V3_GROQ_CONFIG['model']}")

    client = Groq(api_key=groq_api_key)
    rate_limiter = _RateLimiter(max_concurrent, rpm_target, tpm_budget)

    stats = asyncio.run(
        _generate_all(
            clusters, client, rate_limiter, output_path, max_retries_per_step,
        )
    )

    # Phase 3: Write manifest
    manifest = {
        "version": "3.0",
        "output_path": output_path,
        "requested": category_counts,
        "stats": stats,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGeneration complete:")
    print(f"  Generated: {stats['generated']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Validation errors: {stats['validation_errors']}")
    print(f"  By category: {stats['by_category']}")
    print(f"  Total retries: {stats['total_retries']}")
    print(f"  Elapsed: {stats['elapsed_seconds']}s")
    print(f"  Output: {output_path}")
    print(f"  Manifest: {manifest_path}")

    # Optional post-generation validation
    if validate:
        print("\nRunning post-generation validation...")
        error_count = 0
        with open(output_path) as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                errors = _validate_v3_example(record)
                if errors:
                    print(f"  Example {i}: {errors}")
                    error_count += 1
        if error_count == 0:
            print("  All examples passed validation!")
        else:
            print(f"  {error_count} examples failed validation")

    return output_path
