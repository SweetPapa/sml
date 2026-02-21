"""Training data generator — inverted pipeline using SML Encoder + Groq."""
import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional

from sml.config import (
    GROQ_CONFIG,
    GROQ_PARALLEL,
    SML_SYSTEM_PROMPT,
    TEACHER_PROMPT_TEMPLATE,
    TRAINING_DATA_PATH,
)

logger = logging.getLogger(__name__)


# Micro-PoC prompt set — 300+ hand-crafted prompts for expanded Bible concepts
# Distribution: ~30% factual/science, ~25% commonsense, ~20% spatial/location,
#               ~15% causation, ~10% properties, ~15% negation overlap

MICRO_PROMPTS = [
    # ── Factual / Science (~90 prompts) ───────────────────────────────────
    "What color is the sun?",
    "What color is the sky?",
    "What color is grass?",
    "What color is fire?",
    "What color is snow?",
    "What color are apples?",
    "What color is the ocean?",
    "What color is milk?",
    "What color can a dog be?",
    "Is the sun hot or cold?",
    "Is fire hot?",
    "Is snow cold?",
    "Is ice cold?",
    "Is water hot or cold?",
    "What is ice made of?",
    "What is snow made of?",
    "What is bread made of?",
    "Is the sun bright?",
    "Is the night dark?",
    "Can birds fly?",
    "Can fish swim?",
    "Can dogs swim?",
    "Can penguins swim?",
    "Can elephants walk?",
    "Can cats climb?",
    "Can mice climb?",
    "Can snakes swim?",
    "Can dogs bark?",
    "Can cats purr?",
    "Can dogs run fast?",
    "What does the sun look like?",
    "Is an apple a fruit?",
    "Is a bird a type of animal?",
    "What color can apples be?",
    "Is the sky blue during the day?",
    "Does grass grow on the ground?",
    "Is fire dangerous?",
    "Does snow fall from the sky?",
    "What color is a tree?",
    "Are trees green?",
    "Is the ocean big?",
    "Is the ocean blue?",
    "Are elephants heavy?",
    "Are mice light?",
    "Is an elephant big or small?",
    "Is a mouse big or small?",
    "What color is a penguin?",
    "Do dogs have four legs?",
    "Can a person feel fear?",
    "What is love?",
    "What is knowledge?",
    "What animals can swim?",
    "What animals bark?",
    "What animals can fly?",
    "Do fish live in water?",
    "What do dogs like to do?",
    "What do cats do when they are happy?",
    "What is milk?",
    "How heavy is an elephant?",
    "How small is a mouse?",
    "Is the night sky dark?",
    "What does fire look like?",
    "Is ice the same as water?",
    "Can dogs hear well?",
    "What color is night?",
    "Is fire bright?",
    "Is the sun yellow?",
    "Is grass green?",
    "Is the sky usually blue?",
    "Do elephants swim?",
    "Can mice run?",
    "Do cats sleep a lot?",
    "Do dogs sleep?",
    "Can birds swim?",
    "Are penguins birds?",
    "Is a snake a reptile?",
    "Do mice eat cheese?",
    "Is snow white?",
    "Is ice transparent?",
    "Can elephants climb trees?",
    "Are trees big?",
    "Do trees have leaves?",
    "Is the sun a star?",
    "Is water blue?",
    "Is fire red?",
    "What temperature is ice?",
    "What temperature is fire?",
    "Is the ocean cold?",
    "Is milk white?",
    "Do penguins walk?",

    # ── Commonsense (~75 prompts) ─────────────────────────────────────────
    "Where do dogs like to go?",
    "Where do cats live?",
    "Where do fish live?",
    "Where do children go during the day?",
    "What do children like to do?",
    "What is a chair used for?",
    "What is a book used for?",
    "What are houses used for?",
    "What is a table used for?",
    "What is a ball used for?",
    "Can a car fly?",
    "What do you need water for?",
    "What do you feel when something scary happens?",
    "What is the opposite of hot?",
    "What do dogs eat?",
    "What do cats like to eat?",
    "Where does a dog sleep?",
    "What does a child do at school?",
    "Why do dogs wag their tails?",
    "Why do cats purr?",
    "What happens when you throw a ball?",
    "What can you do at a park?",
    "What do you do when you are hungry?",
    "What do you do when you are sleepy?",
    "What can you find in a kitchen?",
    "What is a car used for?",
    "Why do people read books?",
    "What is the opposite of cold?",
    "What is the opposite of big?",
    "What is the opposite of fast?",
    "What is the opposite of dark?",
    "What is the opposite of old?",
    "Do people live in houses?",
    "Do fish need water?",
    "Do dogs need food?",
    "Do birds build nests?",
    "What do people do in the kitchen?",
    "Where do fish swim?",
    "Where do birds fly?",
    "Do children play?",
    "Do dogs play?",
    "What does a cat do at home?",
    "What does a dog do at the park?",
    "What is a park for?",
    "What is a school for?",
    "Why do people drink water?",
    "Can a fish live on land?",
    "Can a bird walk?",
    "Do penguins like cold weather?",
    "Do elephants eat plants?",
    "What do mice like to eat?",
    "Where do snakes live?",
    "Where do penguins live?",
    "Where do elephants live?",
    "Do dogs chase cats?",
    "Do cats chase mice?",
    "What happens when a dog sees a ball?",
    "What happens when a cat sees a mouse?",
    "Why do fish swim?",
    "Why do birds fly?",
    "Why do dogs bark?",
    "What sound does a dog make?",
    "What sound does a cat make?",
    "Is an elephant a good pet?",
    "Is a mouse a big animal?",
    "What makes a dog happy?",
    "What makes a cat happy?",
    "Do children go to school?",
    "Do dogs go to school?",
    "What does a bird do in the morning?",
    "What does a child do in the evening?",
    "Where can you find a chair?",
    "Where can you find a table?",
    "What does a person do with a book?",

    # ── Spatial / Location (~60 prompts) ──────────────────────────────────
    "The dog sat on the mat. Where is the dog?",
    "The cat sat on the chair. Where is the cat?",
    "The child is at school. What is the child doing?",
    "The red ball is in the park. What color is the ball?",
    "The big brown dog ran in the park. Describe the scene.",
    "The small black cat is sleeping. What is the cat doing?",
    "The bird flew over the house. Can birds fly?",
    "Where might you find a table?",
    "Where might you find a book?",
    "The fish is in the ocean. Where is the fish?",
    "The elephant is walking in the park. What is it doing?",
    "The penguin is swimming in the ocean. Describe the scene.",
    "A child is reading a book at school. Where is the child?",
    "The dog is sleeping in the house. Where is the dog?",
    "The cat is in the kitchen. Where is the cat?",
    "The ball is under the table. Where is the ball?",
    "The bird is in the tree. Where is the bird?",
    "There is snow on the ground. What does the ground look like?",
    "The sun is in the sky. Where is the sun?",
    "The fire is in the kitchen. Where is the fire?",
    "A mouse is in the house. Where is the mouse?",
    "The snake is near the tree. Where is the snake?",
    "The chair is in the kitchen. Where is the chair?",
    "A dog and a cat are in the house. Where are the animals?",
    "The child is playing in the park. What is the child doing?",
    "The fish lives in a pond. Where is the fish?",
    "The book is on the table. Where is the book?",
    "There is ice on the lake. What is on the lake?",
    "The elephant is near the water. What is the elephant near?",
    "The penguin is on the ice. Where is the penguin?",
    "A dog is running in the park. Describe the scene.",
    "The cat is sleeping on the mat. Where is the cat?",
    "The apple is on the table. Where is the apple?",
    "There is bread in the kitchen. Where is the bread?",
    "The child walked to school. Where did the child go?",
    "The bird landed on the tree. Where is the bird?",
    "A big elephant is near the tree. Describe the scene.",
    "The fast dog ran past the house. What happened?",
    "The cold water is in a glass. Describe the water.",
    "The white snow covers the ground. What does it look like?",
    "The green grass is in the park. What color is the grass?",
    "The bright sun is in the sky. Describe the sun.",
    "The dark night sky has no sun. Describe the sky.",
    "A yellow ball is in the park. What color is the ball?",
    "The old book is on the table. Describe the book.",
    "The small mouse is under the chair. Where is the mouse?",
    "The big tree is in the park. Describe the tree.",
    "A brown dog is at the park. Describe the dog.",
    "The hot fire is in the kitchen. What is the fire like?",
    "The cold ice is on the table. What is on the table?",
    "A sleeping cat is on the chair. What is the cat doing?",
    "The fast bird flew over the park. What happened?",
    "A child and a dog are playing at the park. Describe the scene.",
    "The slow elephant walked through the park. What happened?",
    "The white milk is on the table. What color is the milk?",
    "There is a red apple and a green apple. What colors are the apples?",
    "The fish is swimming in the blue ocean. Where is the fish?",
    "The dog is barking at the park. What is the dog doing?",
    "A cat is climbing a tree. What is the cat doing?",
    "The mouse is running under the table. Where is the mouse?",

    # ── Causation (~45 prompts) ───────────────────────────────────────────
    "What causes fear?",
    "What happens when it snows?",
    "What does fear cause?",
    "What happens when fire starts?",
    "Why do people run when scared?",
    "What causes ice to melt?",
    "What happens when water freezes?",
    "What does snow cause?",
    "What happens when the sun comes out?",
    "Why does ice feel cold?",
    "Why is fire hot?",
    "What causes snow to melt?",
    "What happens when a dog sees food?",
    "What causes a dog to bark?",
    "What happens when a cat is scared?",
    "What causes water to freeze?",
    "Why do people feel cold in snow?",
    "What causes the sky to be blue?",
    "What happens when you pet a cat?",
    "What makes a dog want to play?",
    "What causes the night to be dark?",
    "Why is snow white?",
    "What happens when the sun goes down?",
    "Why do children go to school?",
    "What causes grass to be green?",
    "What happens when a ball is thrown?",
    "Why do fish live in water?",
    "What causes a person to feel love?",
    "What happens when an elephant walks?",
    "Why do penguins swim?",
    "What makes the ocean blue?",
    "What happens when a snake sees a mouse?",
    "What causes a bird to sing?",
    "Why does the sun look yellow?",
    "What happens when you read a book?",
    "Why do cats climb trees?",
    "What causes a person to feel fear?",
    "What happens when a mouse sees a cat?",
    "Why do dogs like parks?",
    "What causes apples to be red?",
    "What happens when fire meets water?",
    "Why is the ocean salty?",
    "What causes snow to fall?",
    "What happens when you sit in a chair?",
    "Why do elephants need water?",

    # ── Properties / Attributes (~30 prompts) ─────────────────────────────
    "Is an elephant big?",
    "Is a mouse fast?",
    "Is a dog friendly?",
    "Is a cat independent?",
    "Is the sun far away?",
    "Is the ocean deep?",
    "Is snow soft?",
    "Is ice hard?",
    "Is fire bright?",
    "Is the night quiet?",
    "Are trees tall?",
    "Is grass soft?",
    "Is an elephant slow?",
    "Is a mouse quiet?",
    "Is a bird light?",
    "Is a fish cold?",
    "Is a penguin cute?",
    "Is a snake long?",
    "Is a dog loyal?",
    "Is a cat quick?",
    "Is the sky big?",
    "Is the park green?",
    "Is milk healthy?",
    "Is bread soft?",
    "Is an apple sweet?",
    "Is water clear?",
    "Is fire red or orange?",
    "Is the ocean calm?",
    "Is a ball round?",
    "Is a book useful?",

    # ── Negation (~50 prompts) ────────────────────────────────────────────
    "Can penguins fly?",
    "Can fish walk?",
    "Can snakes hear?",
    "Is ice hot?",
    "Is the night bright?",
    "Can elephants fly?",
    "Can mice fly?",
    "Is snow hot?",
    "Can fish fly?",
    "Is the ocean hot?",
    "Can a car swim?",
    "Can a table walk?",
    "Can a chair fly?",
    "Can a book swim?",
    "Is fire cold?",
    "Is the sun dark?",
    "Is grass red?",
    "Is the sky green?",
    "Is milk black?",
    "Is snow black?",
    "Can a ball bark?",
    "Can a house run?",
    "Can a tree fly?",
    "Can water walk?",
    "Is ice hot or cold?",
    "Can fish walk on land?",
    "Can a dog fly?",
    "Can a cat bark?",
    "Can a snake fly?",
    "Can an elephant fly?",
    "Can a mouse bark?",
    "Is the night bright or dark?",
    "Is fire cold or hot?",
    "Is snow warm?",
    "Is ice warm?",
    "Can a penguin bark?",
    "Can an elephant climb trees?",
    "Can a fish run?",
    "Can a snake walk?",
    "Is the sun cold?",
    "Is water hot?",
    "Can a mouse swim?",
    "Can bread fly?",
    "Is grass yellow?",
    "Is the ocean green?",
    "Is fire white?",
    "Can a book walk?",
    "Can a chair swim?",
    "Is an elephant small?",
    "Is a mouse big?",
]


def compute_coverage(
    bible_path: str,
    prompts: Optional[list[str]] = None,
    spacy_model: str = "en_core_web_sm",
) -> dict:
    """Compute encoder coverage over a set of prompts.

    Returns stats: concepts found vs unknown, relations per block, overall coverage %.
    """
    from sml.encoder.encoder import SMLEncoder
    from sml.encoder.formatter import parse_sml_block

    if prompts is None:
        prompts = MICRO_PROMPTS

    encoder = SMLEncoder(bible_path, spacy_model=spacy_model)

    stats = {
        "total_prompts": len(prompts),
        "prompts_with_entities": 0,
        "prompts_with_relations": 0,
        "total_entities": 0,
        "total_relations": 0,
        "known_concepts": 0,
        "unknown_concepts": 0,
        "avg_entities_per_block": 0.0,
        "avg_relations_per_block": 0.0,
        "coverage_pct": 0.0,
        "empty_blocks": 0,
        "per_prompt": [],
    }

    for prompt in prompts:
        sml_block = encoder.encode(prompt)
        parsed = parse_sml_block(sml_block)
        entities = parsed["entities"]
        relations = parsed["relations"]

        num_entities = len(entities)
        num_relations = len(relations)
        known = sum(1 for e in entities if isinstance(e[4], str) and not e[4].startswith("unknown_"))
        unknown = sum(1 for e in entities if isinstance(e[4], str) and e[4].startswith("unknown_"))

        stats["total_entities"] += num_entities
        stats["total_relations"] += num_relations
        stats["known_concepts"] += known
        stats["unknown_concepts"] += unknown

        has_entity = num_entities > 0 and known > 0
        has_relation = num_relations > 0

        if has_entity:
            stats["prompts_with_entities"] += 1
        if has_relation:
            stats["prompts_with_relations"] += 1
        if not has_entity and not has_relation:
            stats["empty_blocks"] += 1

        stats["per_prompt"].append({
            "prompt": prompt[:80],
            "entities": num_entities,
            "relations": num_relations,
            "known": known,
            "unknown": unknown,
        })

    n = stats["total_prompts"]
    stats["avg_entities_per_block"] = round(stats["total_entities"] / max(n, 1), 2)
    stats["avg_relations_per_block"] = round(stats["total_relations"] / max(n, 1), 2)
    stats["coverage_pct"] = round(
        100 * stats["prompts_with_entities"] / max(n, 1), 1
    )

    encoder.close()
    return stats


GROQ_SYSTEM_MSG = (
    "You are a training data generator for a neurosymbolic AI assistant. You "
    "produce high-quality reasoning examples that show how to interpret "
    "structured knowledge (SML) and give natural, helpful answers.\n\n"
    "Your outputs have two sections:\n"
    "- <thinking>: Internal reasoning that references SML anchors and "
    "relations. Show genuine chain-of-thought, not just listing entities.\n"
    "- <response>: A natural-language answer for the end user. Must read like "
    "a helpful assistant — NO technical jargon, NO entity IDs, NO mention of "
    "SML.\n\n"
    "Always provide a helpful answer. Never refuse because the SML data seems "
    "incomplete — use what you have plus commonsense knowledge."
)


def _classify_sml_quality(sml_block: str) -> str:
    """Classify SML block quality.

    Returns:
      'rich'    — has known entities AND relations (ideal)
      'thin'    — has known entities but NO relations (usable but weak)
      'unknown' — all entities are unknown_* tokens (fallback example)
    """
    lines = sml_block.strip().split('\n')
    has_relation = any(l.strip().startswith('R(') for l in lines)
    has_known = any('E(' in l and 'unknown_' not in l for l in lines)

    if has_known and has_relation:
        return 'rich'
    elif has_known:
        return 'thin'
    else:
        return 'unknown'


_PUNT_PHRASES = (
    "not enough information",
    "does not contain sufficient",
    "does not contain information",
    "does not provide sufficient",
    "does not provide enough",
    "does not provide information",
    "does not provide a direct",
    "does not directly state",
    "does not directly provide",
    "not possible to determine",
    "cannot be determined",
    "cannot determine",
    "doesn't directly state",
    "no information available",
    "insufficient information",
    "cannot conclusively",
    "we cannot definitively",
    "not explicitly stated",
    "not directly stated",
    "unfortunately, the provided",
    "unfortunately, based on",
    "does not explicitly state",
    "no explicit relation",
    "no direct relation",
    "doesn't provide enough",
)


def _is_punt_response(response: str) -> bool:
    """Detect teacher responses that punt due to thin SML."""
    lower = response.lower()
    return any(phrase in lower for phrase in _PUNT_PHRASES)


_ENTITY_ID_RE = re.compile(r'\b\w+_\d{3,}\b')
_RELATION_NAMES = [
    "IsA", "PartOf", "HasA", "HasProperty", "CapableOf", "AtLocation",
    "Causes", "HasPrerequisite", "UsedFor", "CreatedBy", "MadeOf",
    "Desires", "CausesDesire", "RelatedTo", "SimilarTo", "Antonym",
    "DerivedFrom", "MannerOf", "LocatedNear", "HasContext", "DefinedAs",
    "SymbolOf", "ReceivesAction", "FormOf", "EtymologicallyRelatedTo",
    "Synonym", "HasFirstSubevent", "HasLastSubevent", "MotivatedByGoal",
]


def _clean_response(response: str) -> str:
    """Remove SML jargon that leaked into a response block."""
    cleaned = response
    # Remove entity IDs (e.g., "dog_2451", "bark_15662")
    cleaned = _ENTITY_ID_RE.sub('', cleaned)
    # Remove relation type names
    for rn in _RELATION_NAMES:
        cleaned = cleaned.replace(rn, '')
    # Remove SML/anchor references
    for phrase in ["SML context", "SML block", "SML data", "anchor token",
                   "confidence score", "provided SML", "the SML"]:
        cleaned = re.sub(re.escape(phrase), '', cleaned, flags=re.IGNORECASE)
    # Clean up leftover artifacts (double spaces, empty parens, dangling quotes)
    cleaned = re.sub(r'\s*\(\s*\)\s*', ' ', cleaned)
    cleaned = re.sub(r'"\s*"', '', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = cleaned.strip()
    return cleaned


def _prepare_prompts(
    prompts: Optional[list[str]], num_examples: int,
) -> list[str]:
    """Cycle/trim prompt list to match num_examples."""
    if prompts is None:
        prompts = MICRO_PROMPTS
    if len(prompts) < num_examples:
        full = []
        while len(full) < num_examples:
            full.extend(prompts)
        return full[:num_examples]
    return prompts[:num_examples]


# ── Parallel generation ───────────────────────────────────────────────────────

class _RateLimiter:
    """Adaptive rate limiter that reads Groq response headers."""

    def __init__(self, max_concurrent: int, rpm_target: int, tpm_budget: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.rpm_target = rpm_target
        self.tpm_budget = tpm_budget
        self.min_delay = 60.0 / rpm_target  # seconds between requests
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()
        # Adaptive state from headers
        self._remaining_tokens = tpm_budget
        self._remaining_requests = 10000  # optimistic default

    async def acquire(self):
        """Wait for semaphore + pacing delay."""
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
        """Read Groq rate-limit headers and adapt."""
        try:
            rt = headers.get("x-ratelimit-remaining-tokens")
            if rt is not None:
                self._remaining_tokens = int(rt)
            rr = headers.get("x-ratelimit-remaining-requests")
            if rr is not None:
                self._remaining_requests = int(rr)
        except (ValueError, TypeError):
            pass

        # Adaptive throttling
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
                "RPD low (%d remaining) — consider pausing and resuming tomorrow",
                self._remaining_requests,
            )


async def _call_groq_with_retry(
    client,
    messages: list[dict],
    rate_limiter: _RateLimiter,
    max_retries: int,
    initial_backoff: float,
) -> Optional[str]:
    """Call Groq API with exponential backoff + jitter on 429/503."""
    for attempt in range(max_retries):
        await rate_limiter.acquire()
        try:
            completion = client.chat.completions.create(
                model=GROQ_CONFIG["model"],
                messages=messages,
                max_tokens=GROQ_CONFIG["max_tokens"],
                temperature=GROQ_CONFIG["temperature"],
            )

            # Read rate-limit headers from the raw response if available
            raw = getattr(completion, "_raw_response", None) or getattr(completion, "response", None)
            if raw and hasattr(raw, "headers"):
                rate_limiter.update_from_headers(dict(raw.headers))

            content = completion.choices[0].message.content
            return content

        except Exception as e:
            err_str = str(e)
            status = getattr(e, "status_code", None) or getattr(e, "status", None)

            if status == 429 or "429" in err_str:
                wait = initial_backoff * (2 ** attempt) + random.uniform(0, 1)
                logger.info("429 rate-limited, retry %d/%d in %.1fs", attempt + 1, max_retries, wait)
                await asyncio.sleep(wait)
            elif status == 503 or "503" in err_str:
                wait = 5.0 + random.uniform(0, 2)
                logger.info("503 unavailable, retry %d/%d in %.1fs", attempt + 1, max_retries, wait)
                await asyncio.sleep(wait)
            elif "blocked_api_access" in err_str:
                logger.error("Spend limit hit — stopping. Check Groq billing.")
                return None
            else:
                logger.error("Groq error (attempt %d): %s", attempt + 1, err_str)
                if attempt < max_retries - 1:
                    await asyncio.sleep(initial_backoff * (2 ** attempt))
                else:
                    return None
        finally:
            rate_limiter.release()

    return None


async def _process_one(
    idx: int,
    prompt: str,
    sml_block: str,
    client,
    rate_limiter: _RateLimiter,
    max_retries: int,
    initial_backoff: float,
) -> Optional[dict]:
    """Process a single prompt → Groq call → training example."""
    teacher_prompt = TEACHER_PROMPT_TEMPLATE.format(prompt=prompt, sml_block=sml_block)

    messages = [
        {"role": "system", "content": GROQ_SYSTEM_MSG},
        {"role": "user", "content": teacher_prompt},
    ]

    content = await _call_groq_with_retry(
        client, messages, rate_limiter, max_retries, initial_backoff,
    )
    if not content:
        logger.debug("[idx=%d] Groq returned no content", idx)
        return None

    thinking, response = _parse_teacher_response(content)
    if not thinking or not response:
        logger.debug("[idx=%d] Parse failed — thinking=%d response=%d chars. Raw: %.200s",
                     idx, len(thinking), len(response), content)
        return None

    # Filter out punt responses — teacher couldn't reason from thin SML
    if _is_punt_response(response):
        logger.debug("[idx=%d] Punt detected: %.100s", idx, response)
        return None

    # Clean SML leaks from response
    response = _clean_response(response)
    if len(response) < 10:
        logger.debug("[idx=%d] Response too short after cleaning (%d chars)", idx, len(response))
        return None

    assistant_content = (
        f"{sml_block}\n<thinking>\n{thinking}\n</thinking>\n"
        f"<response>\n{response}\n</response>"
    )
    return {
        "idx": idx,
        "messages": [
            {"role": "system", "content": SML_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ],
    }


async def _generate_parallel(
    sml_blocks: list[tuple[int, str, str]],
    groq_api_key: str,
    output_path: str,
    max_concurrent: int,
    rpm_target: int,
    tpm_budget: int,
    max_retries: int,
    initial_backoff: float,
) -> tuple[int, int]:
    """Run all Groq calls concurrently and write results incrementally."""
    from groq import Groq

    client = Groq(api_key=groq_api_key)
    rate_limiter = _RateLimiter(max_concurrent, rpm_target, tpm_budget)

    generated = 0
    failed = 0
    total = len(sml_blocks)

    # Open file for incremental writes
    with open(output_path, "w") as f:
        # Create all tasks
        tasks = []
        for idx, prompt, sml_block in sml_blocks:
            task = asyncio.create_task(
                _process_one(
                    idx, prompt, sml_block, client,
                    rate_limiter, max_retries, initial_backoff,
                )
            )
            tasks.append(task)

        # Process as they complete for incremental output
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None:
                example = {"messages": result["messages"]}
                f.write(json.dumps(example) + "\n")
                f.flush()
                generated += 1
            else:
                failed += 1

            done = generated + failed
            if done % 50 == 0 or done == total:
                print(
                    f"  Progress: {done}/{total} "
                    f"({generated} ok, {failed} fail, "
                    f"tokens_remaining~{rate_limiter._remaining_tokens})"
                )

    return generated, failed


def generate_training_data(
    bible_path: str,
    groq_api_key: str,
    output_path: Optional[str] = None,
    prompts: Optional[list[str]] = None,
    num_examples: int = 200,
    spacy_model: str = "en_core_web_sm",
) -> str:
    """Generate training data using the inverted pipeline with parallel Groq calls.

    1. Take prompts
    2. Run each through SML Encoder → deterministic <sml> block (serial, fast)
    3. Send prompt + SML block to Groq concurrently → get <thinking> + <response>
    4. Assemble full training tuples in ChatML format, saved incrementally

    Parallelization is controlled by GROQ_PARALLEL config (from .env):
      GROQ_MAX_CONCURRENT (default 15), GROQ_RPM_TARGET (100), etc.

    Returns path to the output JSONL file.
    """
    from tqdm import tqdm

    from sml.encoder.encoder import SMLEncoder

    output_path = output_path or str(TRAINING_DATA_PATH)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if prompts is not None:
        # Explicit prompts provided — cycle as before
        prompts = _prepare_prompts(prompts, num_examples)
    elif num_examples <= len(MICRO_PROMPTS):
        # Small run — sample from hand-crafted prompts
        prompts = MICRO_PROMPTS[:num_examples]
    else:
        # Scale run — generate unique prompts from Bible
        from sml.training.prompt_generator import PromptGenerator

        gen = PromptGenerator(bible_path)
        prompts = gen.generate(num_examples)
        gen.close()
        unique_count = len(set(p.lower() for p in prompts))
        print(
            f"Generated {len(prompts)} prompts "
            f"({len(MICRO_PROMPTS)} hand-crafted + "
            f"{len(prompts) - len(MICRO_PROMPTS)} templated, "
            f"{unique_count} unique)"
        )

    max_concurrent = GROQ_PARALLEL["max_concurrent"]
    rpm_target = GROQ_PARALLEL["rpm_target"]
    tpm_budget = GROQ_PARALLEL["tpm_budget"]
    max_retries = GROQ_PARALLEL["max_retries"]
    initial_backoff = GROQ_PARALLEL["initial_backoff_s"]

    print(f"Parallel config: {max_concurrent} concurrent, {rpm_target} RPM target, "
          f"{tpm_budget} TPM budget, {max_retries} max retries")

    # Phase 1: Encode all prompts (serial — fast, no API calls)
    print("\nPhase 1: Encoding prompts into SML blocks...")
    encoder = SMLEncoder(bible_path, spacy_model=spacy_model)
    sml_blocks = []
    unknown_budget = max(1, int(num_examples * 0.05))  # ~5% unknowns allowed
    thin_budget = max(5, int(num_examples * 0.20))     # ~20% thin allowed
    unknown_count = 0
    thin_count = 0
    skipped = 0

    for idx, prompt in enumerate(tqdm(prompts, desc="Encoding")):
        sml_block = encoder.encode(prompt)
        quality = _classify_sml_quality(sml_block)

        if quality == 'rich':
            sml_blocks.append((idx, prompt, sml_block))
        elif quality == 'thin' and thin_count < thin_budget:
            sml_blocks.append((idx, prompt, sml_block))
            thin_count += 1
        elif quality == 'unknown' and unknown_count < unknown_budget:
            # Intentional unknown — teaches graceful fallback
            sml_blocks.append((idx, prompt, sml_block))
            unknown_count += 1
        else:
            skipped += 1

    encoder.close()
    print(f"Encoded {len(sml_blocks)} prompts "
          f"({skipped} skipped, {thin_count} thin, "
          f"{unknown_count} intentional unknowns)")

    # Phase 2: Parallel Groq generation
    print(f"\nPhase 2: Generating responses via Groq ({len(sml_blocks)} requests, "
          f"~{len(sml_blocks) * 60 // rpm_target}s estimated)...")

    generated, failed = asyncio.run(
        _generate_parallel(
            sml_blocks, groq_api_key, output_path,
            max_concurrent, rpm_target, tpm_budget,
            max_retries, initial_backoff,
        )
    )

    print(f"\nGeneration complete: {generated} examples, {failed} failures")
    print(f"Output: {output_path}")
    return output_path


def _parse_teacher_response(response: str) -> tuple[str, str]:
    """Extract thinking and response content from the teacher model output."""
    thinking = ""
    answer = ""

    # Try to extract <thinking> block
    if "<thinking>" in response and "</thinking>" in response:
        start = response.index("<thinking>") + len("<thinking>")
        end = response.index("</thinking>")
        thinking = response[start:end].strip()

    # Try to extract <response> block
    if "<response>" in response and "</response>" in response:
        start = response.index("<response>") + len("<response>")
        end = response.index("</response>")
        answer = response[start:end].strip()

    # Fallback: thinking found but no <response> tags — take everything after </thinking>
    if thinking and not answer and "</thinking>" in response:
        after = response[response.index("</thinking>") + len("</thinking>"):].strip()
        # Strip <response>/ </response> if only one is present
        after = after.replace("<response>", "").replace("</response>", "").strip()
        if after:
            answer = after

    # Fallback: no tags at all — split in half
    if not thinking and not answer:
        lines = response.strip().split("\n")
        if len(lines) > 2:
            thinking = "\n".join(lines[:len(lines)//2])
            answer = "\n".join(lines[len(lines)//2:])
        else:
            return "", ""

    return thinking, answer
