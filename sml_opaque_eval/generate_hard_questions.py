#!/usr/bin/env python3
"""Generate Hard SML Opaque Reasoning evaluation dataset (200 questions).

Builds complex 6-12 entity graphs requiring genuine graph traversal,
multi-step inference, conflict resolution, and structural analysis.
All answers are verified by the graph solver.

Usage:
    python sml_opaque_eval/generate_hard_questions.py --no-groq
    python sml_opaque_eval/generate_hard_questions.py --no-groq --verify
    python sml_opaque_eval/generate_hard_questions.py --seed 42 --output custom.jsonl
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

from sml_opaque_eval.graph_engine import Graph

# ── Constants ────────────────────────────────────────────────────────────────

LABELS = ["A", "B", "C", "D"]

CHAIN_RELS = ["IsA", "CapableOf", "Causes", "HasPrerequisite", "PartOf",
              "AtLocation", "UsedFor", "RelatedTo"]

ALL_RELS = CHAIN_RELS + ["HasA", "HasProperty", "SimilarTo"]

NEGATION_RELS = {
    "CapableOf": "NOT_CapableOf",
    "HasProperty": "NOT_HasProperty",
    "IsA": "NOT_IsA",
}


# ── Question assembly helpers (same pattern as generate_questions.py) ────────


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


def pick_distractors(
    all_entities: list[int],
    exclude: set[int],
    count: int,
    rng: random.Random,
) -> list[str]:
    """Pick distractor entity names from the graph, excluding given indices."""
    candidates = [i for i in all_entities if i not in exclude]
    rng.shuffle(candidates)
    return [f"X{c}" for c in candidates[:count]]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1 — Deep Chain Traversal (30 questions)
# 4-5 hop chains with distractor branches. Must follow correct path.
# ═══════════════════════════════════════════════════════════════════════════════


def gen_deep_chain_traversal(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(30):
        g = Graph()

        # Build a 4-5 hop main chain
        hops = rng.choice([4, 5])
        rel_types = [rng.choice(CHAIN_RELS) for _ in range(hops)]
        weights = [round(rng.uniform(0.6, 0.95), 2) for _ in range(hops)]
        chain = g.add_chain(rel_types, weights)

        # Add 2-3 distractor branches from intermediate nodes
        n_branches = rng.randint(2, 3)
        branch_starts = rng.sample(chain[:-1], min(n_branches, len(chain) - 1))
        for bs in branch_starts:
            depth = rng.randint(1, 2)
            d_rels = [rng.choice(CHAIN_RELS) for _ in range(depth)]
            g.add_distractor_branch(bs, depth, d_rels)

        sml_str = g.to_sml()
        start = chain[0]
        end = chain[-1]

        # Verify answer
        endpoint = g.chain_endpoint(start, rel_types)
        assert endpoint == end, f"chain_endpoint mismatch: {endpoint} != {end}"

        all_ents = list(range(g.num_entities()))
        distractors = pick_distractors(all_ents, {start, end}, 3, rng)

        if i % 3 == 0:
            # "What entity does X0 ultimately reach through [rel chain]?"
            rel_desc = " → ".join(rel_types)
            q_text = (f"Following the relation chain ({rel_desc}) starting from X{start}, "
                      f"what entity is reached at the end?")
        elif i % 3 == 1:
            # "How many hops to reach the terminus?"
            q_text = (f"Starting at X{start}, following the chain of relations "
                      f"({', '.join(rel_types)}), what is the final entity?")
        else:
            # "Which entity is at the end of the longest chain from X0?"
            q_text = (f"X{start} begins a chain of {hops} relations. "
                      f"What entity is at the terminus of this chain?")

        choices = [f"X{end}"] + distractors
        questions.append(entry(
            sml_str, q_text, choices, 0, rng,
            category="deep_chain_traversal", difficulty="hard",
            reasoning_type="multi_hop_chain",
            num_entities=g.num_entities(), num_relations=g.num_relations(),
            num_hops=hops,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 2 — Multi-Path Divergence (25 questions)
# Multiple paths from source, must filter by relation type to find correct one.
# ═══════════════════════════════════════════════════════════════════════════════


def gen_multi_path_divergence(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(25):
        g = Graph()
        source = g.add_entity(0.9)

        # Create 3-4 different outgoing paths with different relation types
        n_paths = rng.randint(3, 4)
        path_rels = rng.sample(CHAIN_RELS, n_paths)
        path_endpoints = []

        for rel in path_rels:
            chain_len = rng.randint(2, 3)
            chain_rels = [rel] + [rng.choice(CHAIN_RELS) for _ in range(chain_len - 1)]
            chain_weights = [round(rng.uniform(0.5, 0.95), 2) for _ in range(chain_len)]
            nodes = g.add_chain(chain_rels, chain_weights, start_idx=source)
            path_endpoints.append((rel, nodes[-1]))

        # Pick the query relation type
        query_rel, correct_end = rng.choice(path_endpoints)

        # Verify: following only query_rel from source should reach correct entities
        reachable = g.reachable_from(source, query_rel)
        assert len(reachable) > 0, "No reachable entities via query relation"

        # The first hop via query_rel leads to a specific node
        first_hop = None
        for r in g.relations_from(source, query_rel):
            first_hop = r["tgt"]
            break
        assert first_hop is not None

        sml_str = g.to_sml()
        all_ents = list(range(g.num_entities()))

        # Question: which entity does X0 reach via [specific relation type]?
        wrong_endpoints = [f"X{ep}" for _, ep in path_endpoints if ep != first_hop]
        distractors = wrong_endpoints[:3]
        while len(distractors) < 3:
            more = pick_distractors(all_ents, {source, first_hop}, 3 - len(distractors), rng)
            distractors.extend(more)
        distractors = distractors[:3]

        q_text = (f"X{source} has multiple outgoing relations of different types. "
                  f"Following only {query_rel} relations from X{source}, "
                  f"which entity is directly reached?")

        choices = [f"X{first_hop}"] + distractors
        questions.append(entry(
            sml_str, q_text, choices, 0, rng,
            category="multi_path_divergence", difficulty="hard",
            reasoning_type="relation_type_filter",
            num_entities=g.num_entities(), num_relations=g.num_relations(),
            num_hops=1,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 3 — Confidence Threshold (25 questions)
# "Only following relations above weight X, what's reachable?"
# ═══════════════════════════════════════════════════════════════════════════════


def gen_confidence_threshold(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(25):
        g = Graph()

        # Create 8-10 entities with mixed-weight relations
        n_ents = rng.randint(8, 10)
        for _ in range(n_ents):
            g.add_entity(0.9)

        # Add 6-10 relations with deliberately varied weights
        n_rels = rng.randint(6, 10)
        used_pairs = set()
        for _ in range(n_rels):
            for _ in range(20):  # retry limit
                src = rng.randint(0, n_ents - 1)
                tgt = rng.randint(0, n_ents - 1)
                if src != tgt and (src, tgt) not in used_pairs:
                    break
            else:
                continue
            used_pairs.add((src, tgt))
            # Create a mix: some high, some low
            if rng.random() < 0.4:
                w = round(rng.uniform(0.3, 0.55), 2)
            else:
                w = round(rng.uniform(0.65, 0.95), 2)
            g.add_relation(rng.choice(ALL_RELS[:8]), src, tgt, w)

        # Pick threshold and start node
        threshold = round(rng.choice([0.5, 0.6, 0.7, 0.75]), 2)
        start = rng.randint(0, n_ents - 1)

        # Compute answer
        reachable = g.confidence_filtered_reachable(start, threshold)
        # Also compute unfiltered for distractor
        all_reachable = g.reachable_from(start)

        count_above = len(reachable)
        count_all = len(all_reachable)

        sml_str = g.to_sml()

        if i % 2 == 0:
            # "How many entities are reachable with weight >= threshold?"
            q_text = (f"Starting from X{start}, how many entities can be reached "
                      f"by only following relations with weight >= {threshold}?")

            correct = str(count_above)
            wrong_vals = {str(count_all), str(count_above + 1),
                          str(max(0, count_above - 1))}
            wrong_vals.discard(correct)
            wrong_list = sorted(wrong_vals)[:3]
            while len(wrong_list) < 3:
                wrong_list.append(str(count_above + 2 + len(wrong_list)))
            choices = [correct] + wrong_list[:3]
        else:
            # "Which entities are reachable?"
            if reachable:
                # Pick a reachable entity as correct
                correct_ent = rng.choice(sorted(reachable))
                correct_name = f"X{correct_ent}"
                # Build distractors from non-reachable-above-threshold entities
                exclude = {start, correct_ent}
                candidates = [e for e in range(n_ents) if e not in exclude]
                rng.shuffle(candidates)
                distractors = []
                for c in candidates:
                    name = f"X{c}"
                    if name != correct_name and name not in distractors:
                        distractors.append(name)
                    if len(distractors) >= 3:
                        break

                q_text = (f"Starting from X{start}, only following relations with "
                          f"weight >= {threshold}, which of these entities is reachable?")
                choices = [correct_name] + distractors[:3]
            else:
                # Nothing reachable above threshold
                q_text = (f"Starting from X{start}, only following relations with "
                          f"weight >= {threshold}, how many entities are reachable?")
                choices = ["0", "1", "2", str(max(count_all, 3))]

        questions.append(entry(
            sml_str, q_text, choices, 0, rng,
            category="confidence_threshold", difficulty="hard",
            reasoning_type="weight_filtered_traversal",
            num_entities=g.num_entities(), num_relations=g.num_relations(),
            num_hops=0,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 4 — Transitive Closure (20 questions)
# Count all reachable entities through typed relations.
# ═══════════════════════════════════════════════════════════════════════════════


def gen_transitive_closure(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(20):
        g = Graph()

        # Build a graph with 8-12 entities and typed relation clusters
        n_ents = rng.randint(8, 12)
        for _ in range(n_ents):
            g.add_entity(0.9)

        # Pick 2 relation types: one will form the query, other is distractor
        query_rel = rng.choice(CHAIN_RELS[:6])
        distractor_rel = rng.choice([r for r in CHAIN_RELS[:6] if r != query_rel])

        # Build query-rel chain/network: 4-8 relations
        n_query_rels = rng.randint(4, 8)
        used_pairs = set()
        for _ in range(n_query_rels):
            for _ in range(20):
                src = rng.randint(0, n_ents - 1)
                tgt = rng.randint(0, n_ents - 1)
                if src != tgt and (src, tgt) not in used_pairs:
                    break
            else:
                continue
            used_pairs.add((src, tgt))
            w = round(rng.uniform(0.6, 0.95), 2)
            g.add_relation(query_rel, src, tgt, w)

        # Add distractor relations
        n_dist_rels = rng.randint(3, 6)
        for _ in range(n_dist_rels):
            for _ in range(20):
                src = rng.randint(0, n_ents - 1)
                tgt = rng.randint(0, n_ents - 1)
                if src != tgt and (src, tgt) not in used_pairs:
                    break
            else:
                continue
            used_pairs.add((src, tgt))
            w = round(rng.uniform(0.5, 0.9), 2)
            g.add_relation(distractor_rel, src, tgt, w)

        # Pick start node that has outgoing query rels
        starts_with_query = [e for e in range(n_ents)
                             if g.relations_from(e, query_rel)]
        if not starts_with_query:
            # Ensure at least one start
            g.add_relation(query_rel, 0, 1, 0.8)
            starts_with_query = [0]
        start = rng.choice(starts_with_query)

        # Compute answer: reachable via query_rel only
        reachable = g.reachable_from(start, query_rel)
        count = len(reachable)

        # Also compute all-reachable for distractors
        all_reachable = g.reachable_from(start)
        all_count = len(all_reachable)

        sml_str = g.to_sml()

        q_text = (f"Starting from X{start}, how many distinct entities can be reached "
                  f"by transitively following only {query_rel} relations?")

        correct = str(count)
        wrong = set()
        wrong.add(str(all_count))
        wrong.add(str(count + 1))
        wrong.add(str(max(0, count - 1)))
        wrong.discard(correct)
        wrong_list = sorted(wrong)[:3]
        while len(wrong_list) < 3:
            wrong_list.append(str(count + 2 + len(wrong_list)))
        choices = [correct] + wrong_list[:3]

        questions.append(entry(
            sml_str, q_text, choices, 0, rng,
            category="transitive_closure", difficulty="hard",
            reasoning_type="typed_transitive_count",
            num_entities=g.num_entities(), num_relations=g.num_relations(),
            num_hops=0,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 5 — Inheritance + Negation (25 questions)
# NOT_ overrides on inherited properties.
# ═══════════════════════════════════════════════════════════════════════════════


def gen_inheritance_negation(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(25):
        g = Graph()

        # Build inheritance hierarchy: 2-3 levels
        levels = rng.randint(2, 3)
        hierarchy: list[list[int]] = []

        # Root level
        root = g.add_entity(0.9)
        hierarchy.append([root])

        # Build levels
        for lv in range(1, levels):
            level_nodes = []
            n_children = rng.randint(2, 3)
            for _ in range(n_children):
                child = g.add_entity(0.9)
                parent = rng.choice(hierarchy[lv - 1])
                g.add_relation("IsA", child, parent, round(rng.uniform(0.8, 0.95), 2))
                level_nodes.append(child)
            hierarchy.append(level_nodes)

        # Add capabilities / properties to root and some middle nodes
        n_props = rng.randint(3, 5)
        prop_nodes = []
        for _ in range(n_props):
            p = g.add_entity(0.9)
            prop_nodes.append(p)

        # Root has some properties
        root_props = rng.sample(prop_nodes, min(3, len(prop_nodes)))
        for p in root_props:
            g.add_relation("HasProperty", root, p, round(rng.uniform(0.7, 0.95), 2))

        # Pick a leaf node and negate one inherited property
        leaves = hierarchy[-1]
        query_leaf = rng.choice(leaves)

        negated_prop = rng.choice(root_props)
        g.add_relation("NOT_HasProperty", query_leaf, negated_prop,
                        round(rng.uniform(0.85, 0.99), 2))

        # Add some distractor entities
        for _ in range(rng.randint(1, 3)):
            g.add_entity(0.9)

        sml_str = g.to_sml()

        # Compute effective properties
        effective = g.effective_properties(query_leaf)

        # The negated property should NOT be in effective set
        negated_name = f"X{negated_prop}"
        assert negated_name not in effective, f"NOT_ override failed for {negated_name}"

        # Remaining inherited properties
        inherited = [f"X{p}" for p in root_props if f"X{p}" in effective]

        if i % 3 == 0:
            # "Which property does X_leaf NOT have?"
            q_text = (f"X{query_leaf} inherits from a hierarchy above it. "
                      f"Considering both inherited properties and explicit NOT_ overrides, "
                      f"which property does X{query_leaf} NOT have?")
            non_negated = [f"X{p}" for p in root_props if p != negated_prop]
            distractors = non_negated[:2]
            unreachable = pick_distractors(
                list(range(g.num_entities())),
                {query_leaf, negated_prop} | set(root_props),
                1, rng)
            distractors += unreachable
            distractors = distractors[:3]
            choices = [negated_name] + distractors
        elif i % 3 == 1:
            # "Which property DOES X_leaf retain?"
            if inherited:
                correct = inherited[0]
                distractors = [negated_name]
                more = pick_distractors(
                    list(range(g.num_entities())),
                    {query_leaf} | {int(c[1:]) for c in inherited} | {negated_prop},
                    2, rng)
                distractors += more
                distractors = distractors[:3]
                q_text = (f"X{query_leaf} inherits properties through the IsA hierarchy "
                          f"but has some NOT_ overrides. Which property does "
                          f"X{query_leaf} still have?")
                choices = [correct] + distractors
            else:
                # Edge case: all props negated
                q_text = (f"X{query_leaf} has NOT_ overrides. How many inherited "
                          f"properties does X{query_leaf} retain?")
                choices = ["0", "1", "2", "3"]
        else:
            # "How many effective properties?"
            count = len(effective)
            q_text = (f"X{query_leaf} inherits from its ancestors via IsA relations, "
                      f"but has NOT_HasProperty overrides. How many effective "
                      f"properties does X{query_leaf} have?")
            wrong = {str(count + 1), str(len(root_props)), str(max(0, count - 1))}
            wrong.discard(str(count))
            wrong_list = sorted(wrong)[:3]
            while len(wrong_list) < 3:
                wrong_list.append(str(count + 2 + len(wrong_list)))
            choices = [str(count)] + wrong_list[:3]

        questions.append(entry(
            sml_str, q_text, choices, 0, rng,
            category="inheritance_negation", difficulty="hard",
            reasoning_type="property_override",
            num_entities=g.num_entities(), num_relations=g.num_relations(),
            num_hops=levels,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 6 — Prerequisite Satisfaction (20 questions)
# Can entity achieve goal given blocks?
# ═══════════════════════════════════════════════════════════════════════════════


def gen_prerequisite_satisfaction(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(20):
        g = Graph()

        # Build entity with prereq chain: 3-4 hops
        chain_len = rng.randint(3, 4)
        prereq_rels = ["HasPrerequisite"] * chain_len
        prereq_weights = [round(rng.uniform(0.7, 0.95), 2) for _ in range(chain_len)]
        chain = g.add_chain(prereq_rels, prereq_weights)

        # Add distractor entities
        for _ in range(rng.randint(2, 4)):
            n = g.add_entity(0.9)
            g.add_relation(rng.choice(CHAIN_RELS), n, rng.choice(chain[1:]),
                           round(rng.uniform(0.5, 0.9), 2))

        entity = chain[0]

        if i < 12:
            # Blocked: add NOT_CapableOf for one of the prereqs
            blocked_idx = rng.randint(1, len(chain) - 1)
            blocked_target = chain[blocked_idx]
            g.add_relation("NOT_CapableOf", entity, blocked_target,
                           round(rng.uniform(0.85, 0.99), 2))

            sat, blocked_nodes = g.prerequisite_chain_satisfiable(entity)
            assert not sat, "Expected unsatisfiable"
            assert blocked_target in blocked_nodes

            q_text = (f"X{entity} has a chain of prerequisites. "
                      f"X{entity} also has a NOT_CapableOf relation. "
                      f"Can X{entity} satisfy all its prerequisites?")

            choices = [
                f"No — X{entity} cannot do X{blocked_target} which is required",
                f"Yes — all prerequisites can be met",
                f"Yes — the NOT_CapableOf only blocks direct actions",
                f"Cannot be determined from the data",
            ]
        else:
            # Satisfiable: no blocks
            sat, blocked_nodes = g.prerequisite_chain_satisfiable(entity)
            assert sat, "Expected satisfiable"

            q_text = (f"X{entity} has a chain of {chain_len} prerequisites. "
                      f"Are there any NOT_ relations that block X{entity} from "
                      f"completing this chain?")

            choices = [
                f"No — X{entity} has no NOT_ blocks on its prerequisites",
                f"Yes — X{entity} is blocked at X{chain[1]}",
                f"Yes — the chain is circular and cannot be satisfied",
                f"Cannot be determined from the data",
            ]

        questions.append(entry(
            sml_str=g.to_sml(), q_text=q_text, choices=choices,
            correct_idx=0, rng=rng,
            category="prerequisite_satisfaction", difficulty="hard",
            reasoning_type="prereq_chain_check",
            num_entities=g.num_entities(), num_relations=g.num_relations(),
            num_hops=chain_len,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 7 — Dense Structural (25 questions)
# Hub detection, connectivity, degree analysis in dense graphs.
# ═══════════════════════════════════════════════════════════════════════════════


def gen_dense_structural(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    for i in range(25):
        g = Graph()

        n_ents = rng.randint(8, 12)
        for _ in range(n_ents):
            g.add_entity(0.9)

        # Create a dense graph with 10-15 relations
        n_rels = rng.randint(10, 15)
        used_pairs = set()

        # Ensure one node is clearly a hub
        hub_node = rng.randint(0, n_ents - 1)
        hub_targets = rng.sample([j for j in range(n_ents) if j != hub_node],
                                  min(5, n_ents - 1))
        for t in hub_targets:
            g.add_relation(rng.choice(ALL_RELS[:8]), hub_node, t,
                           round(rng.uniform(0.5, 0.95), 2))
            used_pairs.add((hub_node, t))

        # Fill remaining relations
        remaining = n_rels - len(hub_targets)
        for _ in range(remaining):
            for _ in range(30):
                src = rng.randint(0, n_ents - 1)
                tgt = rng.randint(0, n_ents - 1)
                if src != tgt and (src, tgt) not in used_pairs:
                    break
            else:
                continue
            used_pairs.add((src, tgt))
            g.add_relation(rng.choice(ALL_RELS[:8]), src, tgt,
                           round(rng.uniform(0.4, 0.9), 2))

        # Ensure at least one isolated entity for some questions
        if i % 5 == 4:
            iso = g.add_entity(0.9)  # guaranteed isolated

        sml_str = g.to_sml()

        if i % 5 == 0:
            # Hub detection: which entity has the most connections?
            hubs = g.hub_entities(1)
            hub_idx, hub_deg = hubs[0]
            q_text = ("Which entity in this graph has the highest total degree "
                      "(most connections as source or target)?")
            others = [j for j in range(g.num_entities()) if j != hub_idx]
            rng.shuffle(others)
            distractors = [f"X{o}" for o in others[:3]]
            choices = [f"X{hub_idx}"] + distractors

        elif i % 5 == 1:
            # Degree query: what is the degree of entity X?
            query_ent = rng.randint(0, n_ents - 1)
            deg = g.degree(query_ent)
            q_text = (f"How many total relations (incoming + outgoing) "
                      f"involve X{query_ent}?")
            wrong = {str(deg + 1), str(max(0, deg - 1)), str(deg + 2)}
            wrong.discard(str(deg))
            wrong_list = sorted(wrong)[:3]
            while len(wrong_list) < 3:
                wrong_list.append(str(deg + 3 + len(wrong_list)))
            choices = [str(deg)] + wrong_list[:3]

        elif i % 5 == 2:
            # Reachability count from hub
            reachable = g.reachable_from(hub_node)
            count = len(reachable)
            q_text = (f"How many entities can be reached (directly or transitively) "
                      f"from X{hub_node}?")
            wrong = {str(count + 1), str(n_ents - 1), str(max(0, count - 1))}
            wrong.discard(str(count))
            wrong_list = sorted(wrong)[:3]
            while len(wrong_list) < 3:
                wrong_list.append(str(count + 2 + len(wrong_list)))
            choices = [str(count)] + wrong_list[:3]

        elif i % 5 == 3:
            # Out-degree: which entity has the most outgoing relations?
            out_degrees = [(e, g.out_degree(e)) for e in range(n_ents)]
            out_degrees.sort(key=lambda x: (-x[1], x[0]))
            top_ent, top_out = out_degrees[0]
            q_text = ("Which entity has the most outgoing relations "
                      "(appears as source)?")
            others = [f"X{e}" for e, _ in out_degrees[1:4]]
            while len(others) < 3:
                others.append(f"X{rng.randint(0, n_ents - 1)}")
            choices = [f"X{top_ent}"] + others[:3]

        else:
            # Isolated entity detection
            isolated = g.isolated_entities()
            if isolated:
                iso_ent = min(isolated)
                q_text = ("Which entity in this graph has zero connections "
                          "(neither source nor target of any relation)?")
                connected = [j for j in range(g.num_entities()) if j not in isolated]
                rng.shuffle(connected)
                distractors = [f"X{c}" for c in connected[:3]]
                choices = [f"X{iso_ent}"] + distractors
            else:
                # Fallback: minimum degree
                degs = [(e, g.degree(e)) for e in range(g.num_entities())]
                degs.sort(key=lambda x: (x[1], x[0]))
                min_ent = degs[0][0]
                q_text = ("Which entity has the fewest connections in this graph?")
                others = [f"X{e}" for e, _ in degs[-3:]]
                choices = [f"X{min_ent}"] + others[:3]

        questions.append(entry(
            sml_str, q_text, choices, 0, rng,
            category="dense_structural", difficulty="hard",
            reasoning_type="structural_analysis",
            num_entities=g.num_entities(), num_relations=g.num_relations(),
            num_hops=0,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Category 8 — Contradiction Resolution (30 questions)
# Conflicting positive/negative paths with different weights.
# ═══════════════════════════════════════════════════════════════════════════════


def gen_contradiction_resolution(rng: random.Random) -> list[dict]:
    questions: list[dict] = []

    base_rels = ["CapableOf", "HasProperty"]

    for i in range(30):
        g = Graph()

        # Pick the relation pair
        base_rel = rng.choice(base_rels)
        neg_rel = NEGATION_RELS[base_rel]

        # Build 6-10 entities
        n_ents = rng.randint(6, 10)
        for _ in range(n_ents):
            g.add_entity(0.9)

        source = 0

        if i < 15:
            # Direct contradiction: both positive and negative to same target
            target = rng.randint(1, n_ents - 1)
            pos_w = round(rng.uniform(0.5, 0.95), 2)
            neg_w = round(rng.uniform(0.5, 0.95), 2)

            g.add_relation(base_rel, source, target, pos_w)
            g.add_relation(neg_rel, source, target, neg_w)

            # Add distractor relations
            for _ in range(rng.randint(3, 6)):
                s = rng.randint(0, n_ents - 1)
                t = rng.randint(0, n_ents - 1)
                if s != t:
                    g.add_relation(rng.choice(ALL_RELS[:8]), s, t,
                                   round(rng.uniform(0.4, 0.9), 2))

            pos_paths, neg_paths = g.contradiction_paths(source, target)
            assert len(pos_paths) >= 1 and len(neg_paths) >= 1

            sml_str = g.to_sml()

            if neg_w > pos_w:
                strength_desc = f"the NOT_ relation ({neg_w:.2f}) is stronger"
                correct = (f"The evidence leans toward X{source} NOT "
                           f"having this relation to X{target} (NOT_ weight {neg_w:.2f} > "
                           f"positive weight {pos_w:.2f})")
                wrong1 = (f"The positive relation ({pos_w:.2f}) overrides the "
                          f"NOT_ relation")
                wrong2 = "Both relations cancel out — no conclusion possible"
                wrong3 = f"X{source} definitely has the relation to X{target}"
            else:
                strength_desc = f"the positive relation ({pos_w:.2f}) is stronger"
                correct = (f"The evidence leans toward X{source} having "
                           f"this relation to X{target} (positive weight {pos_w:.2f} > "
                           f"NOT_ weight {neg_w:.2f})")
                wrong1 = (f"The NOT_ relation ({neg_w:.2f}) overrides the "
                          f"positive relation")
                wrong2 = "Both relations cancel out — no conclusion possible"
                wrong3 = f"X{source} definitely does NOT have the relation to X{target}"

            q_text = (f"X{source} has both a {base_rel} relation (weight {pos_w:.2f}) "
                      f"and a {neg_rel} relation (weight {neg_w:.2f}) to X{target}. "
                      f"Based on the relative weights, what is the stronger evidence?")

            choices = [correct, wrong1, wrong2, wrong3]

        else:
            # Indirect contradiction via inheritance
            parent = 1
            child = 0
            prop_target = 2

            g.add_relation("IsA", child, parent, round(rng.uniform(0.8, 0.95), 2))

            # Parent has positive relation to target
            pos_w = round(rng.uniform(0.6, 0.95), 2)
            g.add_relation(base_rel, parent, prop_target, pos_w)

            # Child has negative override
            neg_w = round(rng.uniform(0.7, 0.99), 2)
            g.add_relation(neg_rel, child, prop_target, neg_w)

            # Add distractor entities and relations
            for _ in range(rng.randint(3, 5)):
                s = rng.randint(0, n_ents - 1)
                t = rng.randint(0, n_ents - 1)
                if s != t:
                    g.add_relation(rng.choice(ALL_RELS[:6]), s, t,
                                   round(rng.uniform(0.4, 0.9), 2))

            sml_str = g.to_sml()

            q_text = (f"X{child} is a type of X{parent} (IsA). X{parent} has a "
                      f"{base_rel} relation to X{prop_target} (weight {pos_w:.2f}). "
                      f"But X{child} has a {neg_rel} relation to X{prop_target} "
                      f"(weight {neg_w:.2f}). What applies to X{child}?")

            correct = (f"X{child} does NOT have the {base_rel.lower()} relation to "
                       f"X{prop_target} — the explicit NOT_ override takes precedence")
            wrong1 = (f"X{child} inherits {base_rel} from X{parent}, so the "
                      f"positive relation applies")
            wrong2 = (f"Both apply equally — X{child} both has and doesn't have "
                      f"the relation")
            wrong3 = f"Insufficient information to determine"

            choices = [correct, wrong1, wrong2, wrong3]

        questions.append(entry(
            sml_str, q_text, choices, 0, rng,
            category="contradiction_resolution", difficulty="hard",
            reasoning_type="conflict_resolution",
            num_entities=g.num_entities(), num_relations=g.num_relations(),
            num_hops=2,
        ))

    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════


def verify_questions(questions: list[dict]) -> bool:
    """Re-parse generated JSONL and verify structure/correctness."""
    print("\nVerifying generated questions...")
    errors = 0

    for idx, q in enumerate(questions):
        # Check required fields
        for field in ["question", "choices", "answer", "category",
                      "num_entities", "num_relations"]:
            if field not in q:
                print(f"  Q{idx}: missing field '{field}'")
                errors += 1

        # Check answer index is valid
        if not (0 <= q["answer"] < len(q["choices"])):
            print(f"  Q{idx}: answer index {q['answer']} out of range for {len(q['choices'])} choices")
            errors += 1

        # Check we have exactly 4 choices
        if len(q["choices"]) != 4:
            print(f"  Q{idx}: expected 4 choices, got {len(q['choices'])}")
            errors += 1

        # Check question contains <sml> block
        if "<sml>" not in q["question"] or "</sml>" not in q["question"]:
            print(f"  Q{idx}: missing SML block")
            errors += 1

        # Check no duplicate choices
        if len(set(q["choices"])) != len(q["choices"]):
            print(f"  Q{idx}: duplicate choices detected")
            errors += 1

    if errors == 0:
        print(f"  All {len(questions)} questions passed verification.")
        return True
    else:
        print(f"  {errors} errors found across {len(questions)} questions.")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Groq Rephrasing (optional)
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

        if not rephrased or len(rephrased) > 500:
            return None
        if not rephrased.endswith("?"):
            return None
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
    delay = 60.0 / rpm_target

    rephrased_count = 0
    for idx, q in enumerate(questions):
        full = q["question"]
        parts = full.split("</sml>\n\n", 1)
        if len(parts) != 2:
            continue
        sml_part = parts[0] + "</sml>"
        rest = parts[1]
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
            q["question"] = (
                sml_part + "\n\n" + new_text + "\n" + "\n".join(choice_lines)
            )
            rephrased_count += 1

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(questions)}] rephrased so far: {rephrased_count}")
        time.sleep(delay)

    print(f"Rephrased {rephrased_count}/{len(questions)} questions.")
    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def generate_all(seed: int = 42, use_groq: bool = True) -> list[dict]:
    """Generate all 200 hard questions."""
    rng = random.Random(seed)

    print("Generating hard questions...")
    all_qs: list[dict] = []

    all_qs.extend(gen_deep_chain_traversal(rng))
    print(f"  deep_chain_traversal: {len(all_qs)} questions")

    all_qs.extend(gen_multi_path_divergence(rng))
    print(f"  multi_path_divergence: {len(all_qs)} total")

    all_qs.extend(gen_confidence_threshold(rng))
    print(f"  confidence_threshold: {len(all_qs)} total")

    all_qs.extend(gen_transitive_closure(rng))
    print(f"  transitive_closure: {len(all_qs)} total")

    all_qs.extend(gen_inheritance_negation(rng))
    print(f"  inheritance_negation: {len(all_qs)} total")

    all_qs.extend(gen_prerequisite_satisfaction(rng))
    print(f"  prerequisite_satisfaction: {len(all_qs)} total")

    all_qs.extend(gen_dense_structural(rng))
    print(f"  dense_structural: {len(all_qs)} total")

    all_qs.extend(gen_contradiction_resolution(rng))
    print(f"  contradiction_resolution: {len(all_qs)} total")

    assert len(all_qs) == 200, f"Expected 200 questions, got {len(all_qs)}"

    # Category summary
    cats = {}
    for q in all_qs:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print(f"\nCategory distribution: {json.dumps(cats, indent=2)}")

    # Groq rephrasing
    if use_groq:
        model = os.environ.get(
            "GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
        )
        all_qs = apply_groq_rephrasing(all_qs, model)

    return all_qs


def main():
    parser = argparse.ArgumentParser(
        description="Generate Hard SML Opaque Reasoning evaluation dataset (200 questions)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "sml_hard_reasoning.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-groq",
        action="store_true",
        help="Skip Groq rephrasing — use template questions only",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-parse and verify generated JSONL after writing",
    )
    args = parser.parse_args()

    questions = generate_all(seed=args.seed, use_groq=not args.no_groq)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    print(f"\nWrote {len(questions)} questions to {output}")

    if args.verify:
        # Re-load and verify
        with open(output) as f:
            reloaded = [json.loads(line) for line in f if line.strip()]
        ok = verify_questions(reloaded)
        if not ok:
            sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
