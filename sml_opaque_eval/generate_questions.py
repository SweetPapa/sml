#!/usr/bin/env python3
"""Generate SML Opaque Reasoning evaluation dataset (100 questions).

Algorithmically builds SML blocks and correct answers for guaranteed
correctness.  Optionally uses Groq to rephrase question text for
natural-language variety (--no-groq to skip).

Usage:
    python sml_opaque_eval/generate_questions.py
    python sml_opaque_eval/generate_questions.py --no-groq
    python sml_opaque_eval/generate_questions.py --seed 123
    python sml_opaque_eval/generate_questions.py --output custom.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ── Project setup ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ── Constants ────────────────────────────────────────────────────────────────

LABELS = ["A", "B", "C", "D"]

LOOKUP_RELS = [
    "IsA", "HasA", "PartOf", "AtLocation", "CapableOf", "UsedFor", "HasProperty",
]

RELATION_QUESTION = {
    "IsA":            "what type of entity is {src}?",
    "HasA":           "what does {src} have?",
    "PartOf":         "what is {src} a part of?",
    "AtLocation":     "where is {src} located?",
    "CapableOf":      "what is {src} capable of?",
    "UsedFor":        "what is {src} used for?",
    "HasProperty":    "what property does {src} have?",
    "Causes":         "what does {src} cause?",
    "HasPrerequisite": "what is a prerequisite for {src}?",
    "RelatedTo":      "what is {src} most related to?",
}

# Natural-language version of relation names for question text
RELATION_NATURAL = {
    "RelatedTo": "related",
    "Causes": "a cause of",
    "SimilarTo": "similar",
    "HasProperty": "associated via property with",
    "CapableOf": "capable of something involving",
}

# ── SML construction helpers ────────────────────────────────────────────────


def E(idx: int, conf: float = 0.9) -> str:
    """Entity descriptor line."""
    return f"E(0|0|0|0|X{idx}|0|0|{conf})"


def R(rtype: str, src: int, tgt: int, weight: float = 0.85) -> str:
    """Relation line."""
    return f"R({rtype}|{src}|{tgt}|{weight:.2f}|0|0)"


def sml(ents: list[str], rels: list[str]) -> str:
    """Wrap entity + relation lines in <sml> block."""
    return "<sml>\n" + "\n".join(ents + rels) + "\n</sml>"


# ── Question assembly helpers ────────────────────────────────────────────────


def fmt_question(sml_str: str, q_text: str, choices: list[str]) -> str:
    """Format the full question string (SML block + text + labeled choices)."""
    labeled = [f"{LABELS[i]}) {choices[i]}" for i in range(len(choices))]
    return f"{sml_str}\n\n{q_text}\n" + "\n".join(labeled)


def shuffle_choices(
    choices: list[str], correct_idx: int, rng: random.Random
) -> tuple[list[str], int]:
    """Shuffle choices, return (shuffled_choices, new_correct_idx)."""
    pairs = list(enumerate(choices))
    rng.shuffle(pairs)
    new = [c for _, c in pairs]
    new_idx = next(i for i, (orig, _) in enumerate(pairs) if orig == correct_idx)
    return new, new_idx


def entry(
    sml_str: str,
    q_text: str,
    choices: list[str],
    correct_idx: int,
    rng: random.Random,
    **meta,
) -> dict:
    """Build a complete JSONL entry with shuffled choices."""
    sh, idx = shuffle_choices(choices, correct_idx, rng)
    return {
        "question": fmt_question(sml_str, q_text, sh),
        "choices": sh,
        "answer": idx,
        **meta,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1 — Simple Property Lookup  (20 questions, 2 no-info)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_simple_lookup(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(20):
        rel = LOOKUP_RELS[i % len(LOOKUP_RELS)]
        n = rng.randint(3, 6)
        conf = rng.choice([0.7, 0.8, 0.9, 1.0])
        w = round(rng.uniform(0.4, 1.0), 2)
        ents = [E(j, conf) for j in range(n)]

        # --- no-info (last 2) ---
        if i >= 18:
            tgt = rng.randint(1, n - 1)
            actual_rel = rel
            ask_rel = rng.choice([r for r in LOOKUP_RELS if r != actual_rel])
            rels = [R(actual_rel, 0, tgt, w)]
            s = sml(ents, rels)

            q = f"According to the SML data, {RELATION_QUESTION[ask_rel].format(src='X0')}"
            others = [f"X{j}" for j in range(1, min(n, 4))]
            choices = others + ["Not specified in the SML data"]
            choices = choices[:4]
            while len(choices) < 4:
                choices.append("Cannot be determined")
            # correct = "Not specified"
            not_idx = next(k for k, c in enumerate(choices) if "Not specified" in c)
            questions.append(entry(
                s, q, choices, not_idx, rng,
                category="simple_lookup", difficulty="easy",
                reasoning_type="no_information",
                num_entities=n, num_relations=1, num_hops=0,
            ))
            continue

        # --- regular ---
        tgt = rng.randint(1, n - 1)
        rels = [R(rel, 0, tgt, w)]
        s = sml(ents, rels)

        q = f"According to the SML data, {RELATION_QUESTION[rel].format(src='X0')}"
        others = [j for j in range(1, n) if j != tgt]
        distractors = rng.sample(others, min(3, len(others)))
        choices = [f"X{tgt}"] + [f"X{d}" for d in distractors]
        # Fill up to 4 choices with unique fillers (no duplicates)
        fillers = ["None of the above", "Not specified", "Cannot be determined"]
        filler_idx = 0
        while len(choices) < 4 and filler_idx < len(fillers):
            choices.append(fillers[filler_idx])
            filler_idx += 1
        choices = choices[:4]

        questions.append(entry(
            s, q, choices, 0, rng,
            category="simple_lookup", difficulty="easy",
            reasoning_type="single_relation",
            num_entities=n, num_relations=1, num_hops=1,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 2 — Negation Reasoning  (15 questions, 1 no-info)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_negation(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    neg_pairs = [
        ("CapableOf", "NOT_CapableOf"),
        ("HasProperty", "NOT_HasProperty"),
    ]

    for i in range(15):
        pair = neg_pairs[i % len(neg_pairs)]
        pos_rel, neg_rel = pair
        n = rng.randint(3, 5)
        conf = rng.choice([0.8, 0.9, 1.0])
        ents = [E(j, conf) for j in range(n)]

        # --- no-info (last 1) ---
        if i == 14:
            # only positive relations, question asks about negation
            w = round(rng.uniform(0.6, 1.0), 2)
            rels = [R(pos_rel, 0, 1, w)]
            if n > 2:
                rels.append(R(pos_rel, 0, 2, round(rng.uniform(0.5, 0.9), 2)))
            s = sml(ents, rels)

            q_word = "capable of" if pos_rel == "CapableOf" else "a property of"
            q = f"According to the SML data, what is X0 NOT {q_word}?"
            choices = [f"X{j}" for j in range(1, min(n, 4))]
            choices.append("No negation is specified in the SML data")
            choices = choices[:4]
            while len(choices) < 4:
                choices.append("Cannot be determined")
            not_idx = next(k for k, c in enumerate(choices) if "No negation" in c)
            questions.append(entry(
                s, q, choices, not_idx, rng,
                category="negation", difficulty="medium",
                reasoning_type="no_information",
                num_entities=n, num_relations=len(rels), num_hops=1,
            ))
            continue

        # --- regular ---
        w_pos = round(rng.uniform(0.7, 1.0), 2)
        w_neg = round(rng.uniform(0.7, 1.0), 2)

        # Pick targets: entity 1 = positive, entity 2 = negative (or vice versa)
        pos_tgt = 1
        neg_tgt = 2 if n > 2 else 1

        if neg_tgt == pos_tgt:
            # Only 3 entities available; adjust
            n = max(n, 3)
            ents = [E(j, conf) for j in range(n)]
            neg_tgt = 2

        rels = [
            R(pos_rel, 0, pos_tgt, w_pos),
            R(neg_rel, 0, neg_tgt, w_neg),
        ]
        s = sml(ents, rels)

        # Alternate question types
        if i % 3 == 0:
            # Ask what X0 CANNOT do
            q_word = "capable of" if neg_rel == "NOT_CapableOf" else "a property of"
            q = f"Based on the SML data, what is X0 NOT {q_word}?"
            choices = [f"X{neg_tgt}", f"X{pos_tgt}", f"Both X{pos_tgt} and X{neg_tgt}", "Neither"]
            questions.append(entry(
                s, q, choices, 0, rng,
                category="negation", difficulty="medium",
                reasoning_type="negation_lookup",
                num_entities=n, num_relations=2, num_hops=1,
            ))
        elif i % 3 == 1:
            # Ask what X0 CAN do
            q_word = "capable of" if pos_rel == "CapableOf" else "a property that X0 has"
            q = f"Based on the SML data, what is X0 {q_word}?"
            choices = [f"X{pos_tgt}", f"X{neg_tgt}", f"Both X{pos_tgt} and X{neg_tgt}", "Neither"]
            questions.append(entry(
                s, q, choices, 0, rng,
                category="negation", difficulty="medium",
                reasoning_type="positive_with_negation_distractor",
                num_entities=n, num_relations=2, num_hops=1,
            ))
        else:
            # "Which statement is true?"
            cap = "can do" if pos_rel == "CapableOf" else "has property"
            not_cap = "cannot do" if neg_rel == "NOT_CapableOf" else "does not have property"
            choices = [
                f"X0 {cap} X{pos_tgt} but {not_cap} X{neg_tgt}",
                f"X0 {cap} X{pos_tgt} and X{neg_tgt}",
                f"X0 {not_cap} X{pos_tgt} or X{neg_tgt}",
                f"X0 {not_cap} X{pos_tgt} but {cap} X{neg_tgt}",
            ]
            q = "Based on the SML data, which statement about X0 is true?"
            questions.append(entry(
                s, q, choices, 0, rng,
                category="negation", difficulty="medium",
                reasoning_type="negation_statement",
                num_entities=n, num_relations=2, num_hops=1,
            ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 3 — Multi-Hop / Relation Chains  (20 questions, 2 no-info)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_multi_hop(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    # Chain templates: (rel1, rel2, inference_desc, question_template, correct_template)
    chain_defs = [
        # IsA inheritance (5 Qs)
        ("IsA", "CapableOf",
         "X0 is a type of X1, and X1 is capable of X2.",
         "Based on the SML relations, which of the following can X0 do?",
         "X{end}", "inheritance"),
        ("IsA", "HasProperty",
         "X0 is a type of X1, and X1 has property X2.",
         "Based on the SML relations, which property can be inferred for X0?",
         "X{end}", "inheritance"),
        # Location transitivity (4 Qs)
        ("AtLocation", "PartOf",
         "X0 is at location X1, and X1 is part of X2.",
         "Based on the SML relations, within which larger entity is X0?",
         "X{end}", "transitivity"),
        # Prerequisite chains (4 Qs)
        ("HasPrerequisite", "HasPrerequisite",
         "X0 has prerequisite X1, and X1 has prerequisite X2.",
         "Based on the SML relations, what does X0 indirectly require?",
         "X{end}", "prerequisite_chain"),
        # Causal chains (5 Qs)
        ("Causes", "Causes",
         "X0 causes X1, and X1 causes X2.",
         "Based on the SML relations, what does X0 indirectly cause?",
         "X{end}", "causal_chain"),
    ]

    # Distribution: 5, 2+2(IsA variants), 4(loc), 4(prereq), 5(causal) = ~18 + 2 no-info
    chain_schedule = [0]*3 + [1]*2 + [2]*4 + [3]*4 + [4]*5  # 18 regular
    rng.shuffle(chain_schedule)

    for i in range(20):
        # --- no-info (last 2) ---
        if i >= 18:
            # Chain exists but question asks about unconnected entity
            n = rng.randint(4, 6)
            conf = rng.choice([0.8, 0.9])
            ents = [E(j, conf) for j in range(n)]
            w1 = round(rng.uniform(0.6, 0.95), 2)
            w2 = round(rng.uniform(0.6, 0.95), 2)
            rels = [R("IsA", 0, 1, w1), R("CapableOf", 1, 2, w2)]
            s = sml(ents, rels)
            # Ask about X3 (not in chain)
            distractor = 3 if n > 3 else 2
            q = f"Based on the SML relations, what can X{distractor} do?"
            # Build unique distractor choices
            others = [j for j in range(n) if j != distractor]
            rng.shuffle(others)
            choice_ents = [f"X{others[k]}" for k in range(min(3, len(others)))]
            # Deduplicate
            choice_ents = list(dict.fromkeys(choice_ents))[:3]
            choices = choice_ents + ["Insufficient information"]
            while len(choices) < 4:
                choices.append("Cannot be determined")
            choices = choices[:4]
            ins_idx = choices.index("Insufficient information")
            questions.append(entry(
                s, q, choices, ins_idx, rng,
                category="multi_hop", difficulty="hard",
                reasoning_type="no_information",
                num_entities=n, num_relations=2, num_hops=0,
            ))
            continue

        # --- regular ---
        ci = chain_schedule[i]
        rel1, rel2, _desc, q_template, _ans_template, rtype = chain_defs[ci]

        # 2-hop or 3-hop
        hops = 3 if (i % 5 == 4 and ci in (3, 4)) else 2
        n_chain = hops + 1  # entities in chain
        n_distractors = rng.randint(1, 3)
        n = n_chain + n_distractors
        conf = rng.choice([0.8, 0.9, 1.0])
        ents = [E(j, conf) for j in range(n)]

        rels = []
        for h in range(hops):
            w = round(rng.uniform(0.6, 0.95), 2)
            r = rel1 if h == 0 else rel2
            rels.append(R(r, h, h + 1, w))
        s = sml(ents, rels)

        end = n_chain - 1  # last entity in chain
        correct = f"X{end}"
        q = q_template

        others = [f"X{j}" for j in range(n) if j != end and j != 0]
        dists = rng.sample(others, min(3, len(others)))
        choices = [correct] + dists
        while len(choices) < 4:
            choices.append("Insufficient information")
        choices = choices[:4]

        diff = "medium" if hops == 2 else "hard"
        questions.append(entry(
            s, q, choices, 0, rng,
            category="multi_hop", difficulty=diff,
            reasoning_type=rtype,
            num_entities=n, num_relations=hops, num_hops=hops,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 4 — Weight Comparison  (15 questions, 1 no-info)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_weight_comparison(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    comparison_rels = ["RelatedTo", "Causes", "SimilarTo", "HasProperty", "CapableOf"]

    for i in range(15):
        rel = comparison_rels[i % len(comparison_rels)]
        n = rng.randint(3, 5)
        conf = rng.choice([0.8, 0.9, 1.0])
        ents = [E(j, conf) for j in range(n)]

        # --- no-info (last 1) ---
        if i == 14:
            w = round(rng.uniform(0.5, 0.95), 2)
            rels = [R(rel, 0, 1, w)]
            s = sml(ents, rels)
            q = f"According to the SML data, is X0 more strongly {RELATION_NATURAL.get(rel, rel.lower())} to X1 or X2?"
            choices = [
                f"X1 (weight {w})",
                f"X2",
                "They are equally related",
                "Cannot compare — only one relation exists",
            ]
            questions.append(entry(
                s, q, choices, 3, rng,
                category="weight_comparison", difficulty="medium",
                reasoning_type="no_information",
                num_entities=n, num_relations=1, num_hops=1,
            ))
            continue

        # --- equal weight cases ---
        if i in (11, 12):
            w = round(rng.uniform(0.4, 0.9), 2)
            rels = [R(rel, 0, 1, w), R(rel, 0, 2, w)]
            s = sml(ents, rels)
            q = f"According to the SML data, is X0 more strongly {RELATION_NATURAL.get(rel, rel.lower())} to X1 or X2?"
            choices = [
                f"X1",
                f"X2",
                f"They are equally related (both weight {w})",
                "X0 is not related to either",
            ]
            questions.append(entry(
                s, q, choices, 2, rng,
                category="weight_comparison", difficulty="medium",
                reasoning_type="weight_equal",
                num_entities=n, num_relations=2, num_hops=1,
            ))
            continue

        # --- regular weight comparison ---
        # Determine gap size
        if i < 5:
            # large gap (easy)
            w1 = round(rng.uniform(0.75, 0.98), 2)
            w2 = round(rng.uniform(0.10, 0.35), 2)
            diff = "easy"
        elif i < 10:
            # medium gap
            w1 = round(rng.uniform(0.65, 0.90), 2)
            w2 = round(rng.uniform(0.30, 0.55), 2)
            diff = "medium"
        else:
            # small gap (hard)
            base = round(rng.uniform(0.50, 0.80), 2)
            w1 = round(base + rng.uniform(0.05, 0.12), 2)
            w2 = base
            diff = "hard"

        # Randomize which target has the higher weight
        if rng.random() < 0.5:
            high_tgt, low_tgt = 1, 2
        else:
            high_tgt, low_tgt = 2, 1

        rels = [R(rel, 0, high_tgt, w1), R(rel, 0, low_tgt, w2)]
        s = sml(ents, rels)

        q = f"According to the SML data, is X0 more strongly {RELATION_NATURAL.get(rel, rel.lower())} to X1 or X2?"
        choices = [
            f"X{high_tgt} (weight {w1} vs {w2})",
            f"X{low_tgt} (weight {w2} vs {w1})",
            "They are equally related",
            "X0 is not related to either",
        ]
        questions.append(entry(
            s, q, choices, 0, rng,
            category="weight_comparison", difficulty=diff,
            reasoning_type="weight_comparison",
            num_entities=n, num_relations=2, num_hops=1,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 5 — Counting and Structure  (10 questions, 1 no-info)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_counting(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(10):
        n = rng.randint(3, 7)
        conf = rng.choice([0.8, 0.9, 1.0])
        ents = [E(j, conf) for j in range(n)]

        # Build a random relation set
        n_rels = rng.randint(2, min(6, n * (n - 1) // 2))
        rel_types = ["IsA", "HasA", "CapableOf", "AtLocation", "Causes", "PartOf"]
        rels = []
        used_pairs = set()
        for _ in range(n_rels):
            src = rng.randint(0, n - 1)
            tgt = rng.randint(0, n - 1)
            while tgt == src or (src, tgt) in used_pairs:
                src = rng.randint(0, n - 1)
                tgt = rng.randint(0, n - 1)
            used_pairs.add((src, tgt))
            w = round(rng.uniform(0.3, 1.0), 2)
            rels.append(R(rng.choice(rel_types), src, tgt, w))

        s = sml(ents, rels)

        # ── question sub-types ──

        if i == 9:
            # no-info: ask about entity with no relations
            # Find an isolated entity (or pick one with least connections)
            all_involved = set()
            for r in rels:
                # Parse source and target from relation string
                parts = r.split("|")
                all_involved.add(int(parts[1]))
                all_involved.add(int(parts[2]))
            isolated = [j for j in range(n) if j not in all_involved]
            if isolated:
                ask_ent = rng.choice(isolated)
            else:
                # Make a new entity that's guaranteed isolated
                n += 1
                ents.append(E(n - 1, conf))
                s = sml(ents, rels)
                ask_ent = n - 1

            q = f"How many relations in the SML block involve X{ask_ent}?"
            choices = ["0", "1", "2", "3"]
            questions.append(entry(
                s, q, choices, 0, rng,
                category="counting", difficulty="easy",
                reasoning_type="no_information",
                num_entities=n, num_relations=len(rels), num_hops=0,
            ))
            continue

        if i < 2:
            # Count total entities
            q = "How many entities are defined in the SML block?"
            correct = n
            choices = [str(correct)]
            for delta in [-1, 1, 2]:
                val = correct + delta
                if val > 0 and str(val) not in choices:
                    choices.append(str(val))
            while len(choices) < 4:
                choices.append(str(correct + 3))
            choices = choices[:4]
            c_idx = choices.index(str(correct))
            questions.append(entry(
                s, q, choices, c_idx, rng,
                category="counting", difficulty="easy",
                reasoning_type="count_entities",
                num_entities=n, num_relations=len(rels), num_hops=0,
            ))

        elif i < 4:
            # Count total relations
            q = "How many relations are defined in the SML block?"
            correct = len(rels)
            choices = [str(correct)]
            for delta in [-1, 1, -2]:
                val = correct + delta
                if val >= 0 and str(val) not in choices:
                    choices.append(str(val))
            while len(choices) < 4:
                choices.append(str(correct + 2))
            choices = choices[:4]
            c_idx = choices.index(str(correct))
            questions.append(entry(
                s, q, choices, c_idx, rng,
                category="counting", difficulty="easy",
                reasoning_type="count_relations",
                num_entities=n, num_relations=len(rels), num_hops=0,
            ))

        elif i < 6:
            # Count outgoing relations from X0
            outgoing = sum(1 for r in rels if f"|0|" in r.split("(")[1].split("|")[1] == "0"
                           or r.split("|")[1] == "0")
            # Re-count properly
            outgoing = 0
            for r in rels:
                parts = r.split("|")
                if parts[1] == "0":
                    outgoing += 1

            q = "How many relations in the SML block have X0 as a source?"
            correct = outgoing
            choices = sorted(list(set([str(correct)] + [str(max(0, correct + d)) for d in [-1, 1, 2]])))
            choices = choices[:4]
            while len(choices) < 4:
                choices.append(str(correct + 3))
            c_idx = choices.index(str(correct))
            questions.append(entry(
                s, q, choices, c_idx, rng,
                category="counting", difficulty="medium",
                reasoning_type="count_outgoing",
                num_entities=n, num_relations=len(rels), num_hops=0,
            ))

        elif i < 8:
            # Count unique relation types
            rtypes = set()
            for r in rels:
                rtype = r.split("(")[1].split("|")[0]
                rtypes.add(rtype)
            correct = len(rtypes)

            q = "How many unique relation types appear in the SML block?"
            choices = sorted(list(set([str(correct)] + [str(max(1, correct + d)) for d in [-1, 1, 2]])))
            choices = choices[:4]
            while len(choices) < 4:
                choices.append(str(correct + 3))
            c_idx = choices.index(str(correct))
            questions.append(entry(
                s, q, choices, c_idx, rng,
                category="counting", difficulty="medium",
                reasoning_type="count_unique_types",
                num_entities=n, num_relations=len(rels), num_hops=0,
            ))

        else:  # i == 8
            # Which entity has the most connections (as source or target)?
            counts = {j: 0 for j in range(n)}
            for r in rels:
                parts = r.split("|")
                src = int(parts[1])
                tgt = int(parts[2])
                counts[src] += 1
                counts[tgt] += 1
            most = max(counts, key=lambda k: counts[k])

            q = "Which entity in the SML block has the most connections (as source or target)?"
            others = [j for j in range(n) if j != most]
            dists = rng.sample(others, min(3, len(others)))
            choices = [f"X{most}"] + [f"X{d}" for d in dists]
            while len(choices) < 4:
                choices.append("All entities have equal connections")
            choices = choices[:4]
            questions.append(entry(
                s, q, choices, 0, rng,
                category="counting", difficulty="medium",
                reasoning_type="most_connected",
                num_entities=n, num_relations=len(rels), num_hops=0,
            ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 6 — Composite Reasoning  (20 questions, 2 no-info)
# ═══════════════════════════════════════════════════════════════════════════════


def gen_composite(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(20):
        conf = rng.choice([0.8, 0.9, 1.0])

        # --- no-info (last 2) ---
        if i >= 18:
            n = rng.randint(4, 6)
            ents = [E(j, conf) for j in range(n)]
            w1 = round(rng.uniform(0.6, 0.95), 2)
            w2 = round(rng.uniform(0.6, 0.95), 2)
            rels = [
                R("IsA", 0, 1, w1),
                R("HasProperty", 1, 2, w2),
            ]
            s = sml(ents, rels)
            q = ("The SML block shows relationships between several entities. "
                 f"Based on the SML data, what is X{n-1} capable of?")
            choices = ["X0", "X1", "X2", "Insufficient information"]
            questions.append(entry(
                s, q, choices, 3, rng,
                category="composite", difficulty="hard",
                reasoning_type="no_information",
                num_entities=n, num_relations=2, num_hops=0,
            ))
            continue

        # ── sub-type A: Inheritance + Negation (i 0-3) ──
        if i < 4:
            n = rng.randint(4, 5)
            ents = [E(j, conf) for j in range(n)]
            w1 = round(rng.uniform(0.8, 0.95), 2)
            w2 = round(rng.uniform(0.7, 0.9), 2)
            w3 = round(rng.uniform(0.85, 0.99), 2)

            rels = [
                R("IsA", 0, 1, w1),
                R("CapableOf", 1, 2, w2),
                R("NOT_CapableOf", 0, 2, w3),
            ]
            s = sml(ents, rels)

            q = ("X0 is a type of X1, and X1 can do X2. However, X0 specifically "
                 "cannot do X2. Based on the SML data, what does this tell us?")
            choices = [
                "X0 is an exception — it is a type of X1 that lacks the X2 capability",
                "The SML data is contradictory and no conclusion can be drawn",
                "X0 can do X2 because it inherits from X1",
                "X1 also cannot do X2",
            ]
            questions.append(entry(
                s, q, choices, 0, rng,
                category="composite", difficulty="hard",
                reasoning_type="inheritance_negation",
                num_entities=n, num_relations=3, num_hops=2,
            ))

        # ── sub-type B: Transitivity + Comparison (i 4-7) ──
        elif i < 8:
            n = 5
            ents = [E(j, conf) for j in range(n)]
            w1 = round(rng.uniform(0.6, 0.9), 2)
            w2 = round(rng.uniform(0.6, 0.9), 2)
            w3 = round(rng.uniform(0.6, 0.9), 2)

            rels = [
                R("AtLocation", 0, 1, w1),
                R("AtLocation", 1, 2, w2),
                R("AtLocation", 3, 2, w3),
            ]
            s = sml(ents, rels)

            q = ("X0 is at X1, X1 is at X2, and X3 is also at X2. "
                 "Based on the SML data, are X0 and X3 in the same general area?")
            choices = [
                "Yes — both X0 and X3 are within X2",
                "No — X0 is at X1, which is different from X3's location",
                "Only if X1 and X3 are the same entity",
                "Insufficient information to determine",
            ]
            questions.append(entry(
                s, q, choices, 0, rng,
                category="composite", difficulty="hard",
                reasoning_type="transitivity_comparison",
                num_entities=n, num_relations=3, num_hops=2,
            ))

        # ── sub-type C: Chain + Weight (i 8-11) ──
        elif i < 12:
            n = 4
            ents = [E(j, conf) for j in range(n)]
            w_high = round(rng.uniform(0.80, 0.98), 2)
            w_low = round(rng.uniform(0.10, 0.35), 2)

            rels = [
                R("Causes", 0, 1, w_high),
                R("Causes", 0, 2, w_low),
                R("Causes", 1, 3, round(rng.uniform(0.6, 0.9), 2)),
            ]
            s = sml(ents, rels)

            q = (f"X0 causes X1 (weight {w_high}) and X0 causes X2 (weight {w_low}). "
                 "X1 also causes X3. What is X0 most likely to cause?")
            choices = [
                f"X1 (strongest direct cause at {w_high})",
                f"X2 (weight {w_low})",
                "X3 (via X1)",
                "All are equally likely",
            ]
            questions.append(entry(
                s, q, choices, 0, rng,
                category="composite", difficulty="hard",
                reasoning_type="chain_weight",
                num_entities=n, num_relations=3, num_hops=2,
            ))

        # ── sub-type D: Chain + Negation + Contradiction (i 12-14) ──
        elif i < 15:
            n = 4
            ents = [E(j, conf) for j in range(n)]
            w1 = round(rng.uniform(0.7, 0.95), 2)
            w2 = round(rng.uniform(0.7, 0.95), 2)
            w3 = round(rng.uniform(0.8, 0.99), 2)

            rels = [
                R("HasPrerequisite", 0, 1, w1),
                R("HasPrerequisite", 1, 2, w2),
                R("NOT_CapableOf", 0, 2, w3),
            ]
            s = sml(ents, rels)

            q = ("X0 requires X1, and X1 requires X2. But X0 is NOT capable of X2. "
                 "Based on the SML data, what does this imply?")
            choices = [
                "X0 cannot satisfy its own prerequisites — it needs X2 but cannot do X2",
                "X0 can still satisfy its prerequisites through X1",
                "X1 is also not capable of X2",
                "The prerequisites are optional",
            ]
            questions.append(entry(
                s, q, choices, 0, rng,
                category="composite", difficulty="hard",
                reasoning_type="chain_negation_contradiction",
                num_entities=n, num_relations=3, num_hops=2,
            ))

        # ── sub-type E: Weight + Counting (i 15-17) ──
        else:
            n = rng.randint(4, 6)
            ents = [E(j, conf) for j in range(n)]

            # Multiple relations from X0 with varying weights
            n_rels = rng.randint(3, min(5, n - 1))
            targets = rng.sample(range(1, n), n_rels)
            weights = [round(rng.uniform(0.2, 0.98), 2) for _ in range(n_rels)]
            rels = [R("RelatedTo", 0, t, w) for t, w in zip(targets, weights)]
            s = sml(ents, rels)

            max_w = max(weights)
            max_tgt = targets[weights.index(max_w)]
            above_half = sum(1 for w in weights if w > 0.5)

            q = (f"X0 has {n_rels} relations to other entities with varying weights. "
                 "Which entity is X0 most strongly related to, and how many "
                 "relations have weight above 0.5?")
            correct_str = f"X{max_tgt} is strongest; {above_half} relations above 0.5"
            # Build unique distractors
            wrong_tgt = next((t for t in targets if t != max_tgt), targets[-1])
            distractors = [
                f"X{wrong_tgt} is strongest; {n_rels} relations above 0.5",
                f"X{max_tgt} is strongest; {max(0, above_half - 1)} relations above 0.5",
                "All relations are equally weighted",
            ]
            # Remove any that accidentally match the correct answer
            distractors = [d for d in distractors if d != correct_str]
            # Ensure we have exactly 3 distractors
            fallbacks = [
                f"X{wrong_tgt} is strongest; {above_half} relations above 0.5",
                f"X{max_tgt} is strongest; {above_half + 1} relations above 0.5",
            ]
            for fb in fallbacks:
                if len(distractors) >= 3:
                    break
                if fb != correct_str and fb not in distractors:
                    distractors.append(fb)
            choices = [correct_str] + distractors[:3]
            questions.append(entry(
                s, q, choices, 0, rng,
                category="composite", difficulty="hard",
                reasoning_type="weight_counting",
                num_entities=n, num_relations=n_rels, num_hops=1,
            ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Groq Rephrasing Layer
# ═══════════════════════════════════════════════════════════════════════════════

REPHRASE_SYSTEM = (
    "You rephrase evaluation questions to be more natural and varied. "
    "Return ONLY the rephrased question text — one or two sentences, "
    "ending with a question mark. Do NOT change entity names (X0, X1, etc.), "
    "do NOT change the meaning, and do NOT include answer choices."
)


def rephrase_with_groq(
    client, model: str, question_text: str, category: str
) -> str | None:
    """Ask Groq to rephrase question text. Returns None on failure."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REPHRASE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Category: {category}\n\n"
                        f"Original question: {question_text}\n\n"
                        "Rephrase this question to sound more natural. "
                        "Keep X0, X1, etc. unchanged."
                    ),
                },
            ],
            max_tokens=200,
            temperature=0.5,
        )
        rephrased = resp.choices[0].message.content.strip()

        # Basic validation
        if not rephrased or len(rephrased) > 500:
            return None
        if not rephrased.endswith("?"):
            return None
        # Must still reference relevant entities
        if "X0" in question_text and "X0" not in rephrased:
            return None
        return rephrased
    except Exception as exc:
        print(f"  Groq rephrase failed: {exc}")
        return None


def apply_groq_rephrasing(questions: list[dict], model: str) -> list[dict]:
    """Optionally rephrase question text via Groq for natural variety."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("GROQ_API_KEY not set — skipping rephrasing, using templates only.")
        return questions

    client = Groq(api_key=api_key)
    print(f"Rephrasing {len(questions)} questions via Groq ({model})...")

    rpm_target = int(os.environ.get("GROQ_RPM_TARGET", "100"))
    delay = 60.0 / rpm_target  # seconds between calls

    rephrased_count = 0
    for idx, q in enumerate(questions):
        # Extract just the question text (between SML block and choices)
        full = q["question"]
        # Split: sml block, then question+choices
        parts = full.split("</sml>\n\n", 1)
        if len(parts) != 2:
            continue
        sml_part = parts[0] + "</sml>"
        rest = parts[1]
        # Split question text from choices
        lines = rest.split("\n")
        q_lines = []
        choice_lines = []
        for line in lines:
            if line and line[0] in "ABCD" and line[1:2] == ")":
                choice_lines.append(line)
            else:
                q_lines.append(line)
        q_text = "\n".join(q_lines).strip()

        new_text = rephrase_with_groq(client, model, q_text, q["category"])
        if new_text:
            # Rebuild the full question
            q["question"] = (
                sml_part + "\n\n" + new_text + "\n" + "\n".join(choice_lines)
            )
            rephrased_count += 1

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(questions)}] rephrased so far: {rephrased_count}")
        time.sleep(delay)

    print(f"Rephrased {rephrased_count}/{len(questions)} questions.")
    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def generate_all(seed: int = 42, use_groq: bool = True) -> list[dict]:
    """Generate all 100 questions."""
    rng = random.Random(seed)

    print("Generating questions...")
    all_qs: list[dict] = []
    all_qs.extend(gen_simple_lookup(rng))
    print(f"  simple_lookup: {len(all_qs)} questions")
    all_qs.extend(gen_negation(rng))
    print(f"  negation: {len(all_qs)} total")
    all_qs.extend(gen_multi_hop(rng))
    print(f"  multi_hop: {len(all_qs)} total")
    all_qs.extend(gen_weight_comparison(rng))
    print(f"  weight_comparison: {len(all_qs)} total")
    all_qs.extend(gen_counting(rng))
    print(f"  counting: {len(all_qs)} total")
    all_qs.extend(gen_composite(rng))
    print(f"  composite: {len(all_qs)} total")

    assert len(all_qs) == 100, f"Expected 100 questions, got {len(all_qs)}"

    # Category summary
    cats = {}
    for q in all_qs:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"\nCategory distribution: {json.dumps(cats, indent=2)}")

    no_info = sum(1 for q in all_qs if q.get("reasoning_type") == "no_information")
    print(f"No-information questions: {no_info}")

    # Groq rephrasing
    if use_groq:
        model = os.environ.get(
            "GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
        )
        all_qs = apply_groq_rephrasing(all_qs, model)

    return all_qs


def main():
    parser = argparse.ArgumentParser(
        description="Generate SML Opaque Reasoning evaluation dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "sml_opaque_reasoning.jsonl"),
        help="Output JSONL path (default: sml_opaque_eval/sml_opaque_reasoning.jsonl)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-groq",
        action="store_true",
        help="Skip Groq rephrasing — use template questions only",
    )
    args = parser.parse_args()

    questions = generate_all(seed=args.seed, use_groq=not args.no_groq)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    print(f"\nWrote {len(questions)} questions to {output}")
    print("Done!")


if __name__ == "__main__":
    main()
