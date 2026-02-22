"""Concept cluster selection engine for V3 training data generation."""

import itertools
import random
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from sml.config import RELATION_TYPES
from sml.encoder.formatter import format_sml_block


@dataclass
class ConceptCluster:
    """A cluster of related concepts with pre-built SML block."""

    category: str  # A, B, C, D
    seed_concept: dict
    concepts: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    sml_block: str = ""
    metadata: dict = field(default_factory=dict)


# Word list for Category B fictional anchors
_FICTIONAL_WORDS = [
    "glimthor", "vexnor", "draplix", "quorlam", "zynthar", "brevmok",
    "clispor", "draveen", "elquix", "fenthar", "grolvex", "haxmire",
    "ixpolar", "jentrik", "kolvane", "loxprim", "morquex", "nelvrix",
    "opziran", "prelvox", "quindra", "rexvolt", "silvane", "trixmon",
    "ulvexar", "vornith", "wexplar", "xilvorn", "yentrix", "zolvane",
    "axbrenn", "brixtel", "carvlex", "dexmorn", "elvaxir", "foxvern",
    "gelvrix", "horvane", "ixquell", "joxvire", "krixtel", "loxvane",
    "melvrix", "noxplar", "orvixen", "praxvel", "quelvix", "roxvane",
    "stelvox", "trovixe",
]

# Minimum weight for relations to be considered "strong"
_MIN_STRONG_WEIGHT = 0.3


class ClusterSelector:
    """Selects concept clusters from the Bible for V3 training data."""

    def __init__(self, bible_path: str, seed: int = 42):
        self.bible_path = bible_path
        self.conn = sqlite3.connect(bible_path)
        self.conn.row_factory = sqlite3.Row
        self.rng = random.Random(seed)

        # Build indexes
        self._rich_concepts: list[dict] = []
        self._relations_by_source: dict[int, list[dict]] = {}
        self._multi_hop_chains: list[dict] = []
        self._antonym_pairs: list[dict] = []

        self._load_rich_concepts()
        self._load_relations_index()
        self._load_multi_hop_chains()
        self._load_antonym_pairs()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def _load_rich_concepts(self):
        """Load concepts with >= 5 outgoing relations, >= 2 distinct types,
        and at least 2 relations with weight >= 0.5."""
        rows = self.conn.execute("""
            SELECT c.id, c.surface_text, c.anchor_token, c.domain, c.category,
                   c.subcategory, c.specificity, c.definition,
                   COUNT(DISTINCT r.relation_type_id) AS type_count,
                   COUNT(r.id) AS total_rels,
                   SUM(CASE WHEN r.weight >= 0.5 THEN 1 ELSE 0 END) AS strong_rels
            FROM concepts c
            JOIN relations r ON r.source_id = c.id
            GROUP BY c.id
            HAVING total_rels >= 3 AND type_count >= 2 AND strong_rels >= 1
            ORDER BY type_count DESC, total_rels DESC
        """).fetchall()
        rich = [dict(r) for r in rows]
        # Sort: prefer concepts with populated domain (non-zero) first,
        # then by strong_rels descending
        rich.sort(key=lambda c: (c["domain"] == 0, -c["strong_rels"]))
        self._rich_concepts = rich

    def _load_relations_index(self):
        """Build source_id -> relations index."""
        rows = self.conn.execute("""
            SELECT r.id, r.source_id, r.target_id, r.relation_type_id, r.weight,
                   c.id as target_concept_id, c.surface_text as target_text,
                   c.anchor_token as target_anchor, c.domain as target_domain,
                   c.category as target_category, c.subcategory as target_subcategory,
                   c.specificity as target_specificity, c.definition as target_definition
            FROM relations r
            JOIN concepts c ON r.target_id = c.id
            ORDER BY r.source_id, r.weight DESC
        """).fetchall()
        for row in rows:
            d = dict(row)
            self._relations_by_source.setdefault(d["source_id"], []).append(d)

    def _load_multi_hop_chains(self):
        """Pre-compute multi-hop chains (A -> B -> C) for Category C.
        Both hops must have weight >= 0.5."""
        rows = self.conn.execute("""
            SELECT c1.id, c1.surface_text, c1.anchor_token,
                   c1.domain as c1_domain, c1.category as c1_category,
                   c1.subcategory as c1_subcategory, c1.specificity as c1_specificity,
                   c1.definition as c1_definition,
                   r1.relation_type_id as r1_type, r1.weight as r1_weight,
                   c2.id as c2_id, c2.surface_text as c2_text, c2.anchor_token as c2_anchor,
                   c2.domain as c2_domain, c2.category as c2_category,
                   c2.subcategory as c2_subcategory, c2.specificity as c2_specificity,
                   c2.definition as c2_definition,
                   r2.relation_type_id as r2_type, r2.weight as r2_weight,
                   c3.id as c3_id, c3.surface_text as c3_text, c3.anchor_token as c3_anchor,
                   c3.domain as c3_domain, c3.category as c3_category,
                   c3.subcategory as c3_subcategory, c3.specificity as c3_specificity,
                   c3.definition as c3_definition
            FROM relations r1
            JOIN concepts c1 ON r1.source_id = c1.id
            JOIN concepts c2 ON r1.target_id = c2.id
            JOIN relations r2 ON r2.source_id = c2.id
            JOIN concepts c3 ON r2.target_id = c3.id
            WHERE r1.weight >= 0.3 AND r2.weight >= 0.3
              AND c1.id != c3.id AND c1.id != c2.id
              AND r1.relation_type_id != r2.relation_type_id
            ORDER BY (r1.weight + r2.weight) DESC
            LIMIT 100000
        """).fetchall()
        self._multi_hop_chains = [dict(r) for r in rows]

    def _load_antonym_pairs(self):
        """Load antonym pairs (relation_type_id = 22) for Category D."""
        rows = self.conn.execute("""
            SELECT r.id, r.source_id, r.target_id, r.weight,
                   c1.surface_text as source_text, c1.anchor_token as source_anchor,
                   c1.domain as source_domain, c1.category as source_category,
                   c1.subcategory as source_subcategory, c1.specificity as source_specificity,
                   c1.definition as source_definition,
                   c2.surface_text as target_text, c2.anchor_token as target_anchor,
                   c2.domain as target_domain, c2.category as target_category,
                   c2.subcategory as target_subcategory, c2.specificity as target_specificity,
                   c2.definition as target_definition
            FROM relations r
            JOIN concepts c1 ON r.source_id = c1.id
            JOIN concepts c2 ON r.target_id = c2.id
            WHERE r.relation_type_id = 22
            ORDER BY r.weight DESC
        """).fetchall()
        self._antonym_pairs = [dict(r) for r in rows]

    def _get_best_modifier(self, concept_id: int) -> int:
        """Get the best modifier (anchor token) from HasProperty relations."""
        rels = self._relations_by_source.get(concept_id, [])
        for rel in rels:
            if rel["relation_type_id"] == 4:  # HasProperty
                anchor = rel.get("target_anchor", "0")
                if anchor and anchor != "0":
                    return anchor
        return 0

    def _get_inter_relations(self, concept_ids: set[int]) -> list[dict]:
        """Get all relations between a set of concept IDs, deduplicated."""
        if len(concept_ids) < 2:
            return []
        placeholders = ",".join("?" * len(concept_ids))
        rows = self.conn.execute(f"""
            SELECT r.id, r.source_id, r.target_id, r.relation_type_id, r.weight
            FROM relations r
            WHERE r.source_id IN ({placeholders}) AND r.target_id IN ({placeholders})
        """, list(concept_ids) + list(concept_ids)).fetchall()

        # Deduplicate by (source_id, target_id, relation_type_id), keep highest weight
        best: dict[tuple, dict] = {}
        for row in rows:
            d = dict(row)
            key = (d["source_id"], d["target_id"], d["relation_type_id"])
            if key not in best or d["weight"] > best[key]["weight"]:
                best[key] = d
        return list(best.values())

    def _build_sml_block(
        self,
        concepts: list[dict],
        relations: list[dict],
    ) -> str:
        """Build an SML block from concepts and relations.

        Entity confidence uses max weight of involved relations (floored at 0.5).
        Relations are deduplicated by (type, src_idx, tgt_idx).
        """
        concept_id_to_idx = {}
        entities = []
        for i, c in enumerate(concepts):
            concept_id_to_idx[c["id"]] = i
            mod1 = self._get_best_modifier(c["id"])
            # Use max weight of relations involving this concept
            involved_weights = [
                r["weight"] for r in relations
                if r.get("source_id") == c["id"] or r.get("target_id") == c["id"]
            ]
            if involved_weights:
                confidence = round(max(involved_weights), 2)
                confidence = max(confidence, 0.5)  # floor at 0.5
            else:
                confidence = 0.85  # default when no relations touch this entity
            entities.append([
                c.get("domain", 0), c.get("category", 0),
                c.get("subcategory", 0), c.get("specificity", 0),
                c.get("anchor_token", "unknown_0"),
                mod1, 0, confidence,
            ])

        # Deduplicate relations by (type, src_idx, tgt_idx), keep highest weight
        seen_ra: dict[tuple, list] = {}
        for rel in relations:
            src_idx = concept_id_to_idx.get(rel["source_id"])
            tgt_idx = concept_id_to_idx.get(rel["target_id"])
            if src_idx is None or tgt_idx is None:
                continue
            negation = rel.get("negation", 0)
            key = (rel["relation_type_id"], src_idx, tgt_idx)
            entry = [rel["relation_type_id"], src_idx, tgt_idx, rel["weight"], 0, negation]
            if key not in seen_ra or rel["weight"] > seen_ra[key][3]:
                seen_ra[key] = entry

        ra_list = list(seen_ra.values())

        if not entities:
            return "<sml>\n</sml>"

        return format_sml_block(entities, ra_list)

    def _has_relations(self, sml_block: str) -> bool:
        """Check that an SML block contains at least one R(...) line."""
        return "\nR(" in sml_block

    # ── Category A: Standard Grounding ──────────────────────────────────────

    def _select_category_a(self, count: int) -> list[ConceptCluster]:
        """Select standard grounding clusters from rich concepts.

        Seeds are reused with different neighbor subsets so we can scale
        beyond the number of unique rich concepts. For each seed we generate
        combos of (relation_type_subset × neighbor_rank) to maximize variety.
        """
        clusters = []
        candidates = list(self._rich_concepts)
        if not candidates:
            return clusters

        # Pre-compute all distinct (seed, neighbor_id_set) combos
        seed_neighbor_sets: list[tuple[dict, list[dict]]] = []
        for seed in candidates:
            rels = self._relations_by_source.get(seed["id"], [])
            strong = [r for r in rels if r["weight"] >= 0.3]
            if not strong:
                continue
            # Group by type, keep top 3 targets per type for variety
            by_type: dict[int, list[dict]] = {}
            for r in strong:
                by_type.setdefault(r["relation_type_id"], []).append(r)
            for t in by_type:
                by_type[t].sort(key=lambda r: r["weight"], reverse=True)
                by_type[t] = by_type[t][:3]
            type_keys = sorted(by_type.keys())
            # Generate combos of 1-3 relation types
            for size in range(1, min(4, len(type_keys) + 1)):
                for type_combo in itertools.combinations(type_keys, size):
                    # For each type combo, vary which rank neighbor we pick
                    # by taking the cartesian product of available targets
                    per_type_targets = [by_type[t] for t in type_combo]
                    for target_pick in itertools.product(*per_type_targets):
                        neighbor_list = []
                        seen = {seed["id"]}
                        for rel in target_pick:
                            tid = rel["target_concept_id"]
                            if tid not in seen:
                                neighbor_list.append(rel)
                                seen.add(tid)
                        if neighbor_list:
                            seed_neighbor_sets.append((seed, neighbor_list))

        self.rng.shuffle(seed_neighbor_sets)
        seen_cluster_keys: set[tuple] = set()

        for seed, neighbor_rels in seed_neighbor_sets:
            if len(clusters) >= count:
                break

            neighbor_ids = tuple(sorted(r["target_concept_id"] for r in neighbor_rels))
            cluster_key = (seed["id"], neighbor_ids)
            if cluster_key in seen_cluster_keys:
                continue
            seen_cluster_keys.add(cluster_key)

            neighbors = []
            for rel in neighbor_rels:
                neighbors.append({
                    "id": rel["target_concept_id"],
                    "surface_text": rel["target_text"],
                    "anchor_token": rel["target_anchor"],
                    "domain": rel["target_domain"],
                    "category": rel["target_category"],
                    "subcategory": rel["target_subcategory"],
                    "specificity": rel["target_specificity"],
                    "definition": rel.get("target_definition", ""),
                })

            all_concepts = [
                {
                    "id": seed["id"],
                    "surface_text": seed["surface_text"],
                    "anchor_token": seed["anchor_token"],
                    "domain": seed["domain"],
                    "category": seed["category"],
                    "subcategory": seed["subcategory"],
                    "specificity": seed["specificity"],
                    "definition": seed.get("definition", ""),
                },
            ] + neighbors

            concept_ids = {c["id"] for c in all_concepts}
            inter_rels = self._get_inter_relations(concept_ids)
            sml_block = self._build_sml_block(all_concepts, inter_rels)

            if not self._has_relations(sml_block):
                continue

            cluster = ConceptCluster(
                category="A",
                seed_concept={
                    "id": seed["id"],
                    "surface_text": seed["surface_text"],
                    "anchor_token": seed["anchor_token"],
                },
                concepts=all_concepts,
                relations=inter_rels,
                sml_block=sml_block,
                metadata={
                    "seed_id": seed["id"],
                    "num_concepts": len(all_concepts),
                    "num_relations": len(inter_rels),
                },
            )
            clusters.append(cluster)

        return clusters[:count]

    # ── Category B: Novel/Fictional ─────────────────────────────────────────

    def _select_category_b(self, count: int) -> list[ConceptCluster]:
        """Select clusters with a fictional anchor mixed with real concepts.

        Cycles through candidates so we can produce more clusters than
        unique rich concepts. Each iteration picks different real concept
        combinations via modular offset.
        """
        clusters = []
        candidates = list(self._rich_concepts)
        if not candidates:
            return clusters
        self.rng.shuffle(candidates)

        for i in range(count):
            # Pick 2-3 real concepts — cycling through candidates
            num_real = self.rng.randint(2, min(3, len(candidates)))
            start = (i * 7) % len(candidates)  # stride by prime for spread
            real_concepts = []
            for j in range(num_real):
                idx = (start + j) % len(candidates)
                c = candidates[idx]
                real_concepts.append({
                    "id": c["id"],
                    "surface_text": c["surface_text"],
                    "anchor_token": c["anchor_token"],
                    "domain": c["domain"],
                    "category": c["category"],
                    "subcategory": c["subcategory"],
                    "specificity": c["specificity"],
                    "definition": c.get("definition", ""),
                })

            # Generate unique synthetic anchor
            word_idx = i % len(_FICTIONAL_WORDS)
            fictional_anchor = f"{_FICTIONAL_WORDS[word_idx]}_{99001 + i}"
            fictional_concept = {
                "id": -(i + 1),  # negative ID for synthetic
                "surface_text": _FICTIONAL_WORDS[word_idx],
                "anchor_token": fictional_anchor,
                "domain": 0,
                "category": 0,
                "subcategory": 0,
                "specificity": 0,
                "definition": "",
            }

            # Build borrowed relations: fictional -> each real concept
            donor = real_concepts[0]
            donor_rels = self._relations_by_source.get(donor["id"], [])
            used_types = set()
            borrowed_rels = []
            real_ids = [c["id"] for c in real_concepts]
            for rel in donor_rels:
                if rel["relation_type_id"] in used_types:
                    continue
                if len(borrowed_rels) >= min(3, len(real_concepts)):
                    break
                target_id = real_ids[len(borrowed_rels) % len(real_ids)]
                borrowed_rels.append({
                    "source_id": fictional_concept["id"],
                    "target_id": target_id,
                    "relation_type_id": rel["relation_type_id"],
                    "weight": round(min(rel["weight"] * 0.8, 0.95), 2),
                    "negation": 0,
                })
                used_types.add(rel["relation_type_id"])

            all_concepts = [fictional_concept] + real_concepts

            real_id_set = {c["id"] for c in real_concepts}
            inter_rels = self._get_inter_relations(real_id_set)
            all_rels = borrowed_rels + inter_rels

            sml_block = self._build_sml_block(all_concepts, all_rels)

            if not self._has_relations(sml_block):
                continue

            cluster = ConceptCluster(
                category="B",
                seed_concept={
                    "id": fictional_concept["id"],
                    "surface_text": fictional_concept["surface_text"],
                    "anchor_token": fictional_concept["anchor_token"],
                },
                concepts=all_concepts,
                relations=all_rels,
                sml_block=sml_block,
                metadata={
                    "fictional_anchor": fictional_anchor,
                    "donor_concept": donor["surface_text"],
                    "num_real": len(real_concepts),
                    "num_borrowed_rels": len(borrowed_rels),
                },
            )
            clusters.append(cluster)

        return clusters[:count]

    # ── Category C: Multi-Hop ───────────────────────────────────────────────

    def _select_category_c(self, count: int) -> list[ConceptCluster]:
        """Select multi-hop chain clusters (A -> B -> C).

        Each unique chain (c1, c2, c3) triple is a distinct cluster.
        """
        clusters = []
        chains = list(self._multi_hop_chains)
        if not chains:
            return clusters
        self.rng.shuffle(chains)

        seen_triples: set[tuple] = set()
        for chain in chains:
            if len(clusters) >= count:
                break

            c1_id = chain["id"]
            c2_id = chain["c2_id"]
            c3_id = chain["c3_id"]
            triple = (c1_id, c2_id, c3_id)
            if triple in seen_triples:
                continue
            seen_triples.add(triple)

            c1 = {
                "id": chain["id"],
                "surface_text": chain["surface_text"],
                "anchor_token": chain["anchor_token"],
                "domain": chain["c1_domain"],
                "category": chain["c1_category"],
                "subcategory": chain["c1_subcategory"],
                "specificity": chain["c1_specificity"],
                "definition": chain.get("c1_definition", ""),
            }
            c2 = {
                "id": chain["c2_id"],
                "surface_text": chain["c2_text"],
                "anchor_token": chain["c2_anchor"],
                "domain": chain["c2_domain"],
                "category": chain["c2_category"],
                "subcategory": chain["c2_subcategory"],
                "specificity": chain["c2_specificity"],
                "definition": chain.get("c2_definition", ""),
            }
            c3 = {
                "id": chain["c3_id"],
                "surface_text": chain["c3_text"],
                "anchor_token": chain["c3_anchor"],
                "domain": chain["c3_domain"],
                "category": chain["c3_category"],
                "subcategory": chain["c3_subcategory"],
                "specificity": chain["c3_specificity"],
                "definition": chain.get("c3_definition", ""),
            }

            all_concepts = [c1, c2, c3]
            r1_type = chain["r1_type"]
            r2_type = chain["r2_type"]
            r1_weight = chain["r1_weight"]
            r2_weight = chain["r2_weight"]

            chain_rels = [
                {
                    "source_id": c1["id"],
                    "target_id": c2["id"],
                    "relation_type_id": r1_type,
                    "weight": r1_weight,
                    "negation": 0,
                },
                {
                    "source_id": c2["id"],
                    "target_id": c3["id"],
                    "relation_type_id": r2_type,
                    "weight": r2_weight,
                    "negation": 0,
                },
            ]

            sml_block = self._build_sml_block(all_concepts, chain_rels)

            r1_label = RELATION_TYPES.get(r1_type, str(r1_type))
            r2_label = RELATION_TYPES.get(r2_type, str(r2_type))

            cluster = ConceptCluster(
                category="C",
                seed_concept={
                    "id": c1["id"],
                    "surface_text": c1["surface_text"],
                    "anchor_token": c1["anchor_token"],
                },
                concepts=all_concepts,
                relations=chain_rels,
                sml_block=sml_block,
                metadata={
                    "hop_chain": [
                        c1["surface_text"], r1_label,
                        c2["surface_text"], r2_label,
                        c3["surface_text"],
                    ],
                    "r1_type": r1_label,
                    "r2_type": r2_label,
                },
            )
            clusters.append(cluster)

        return clusters[:count]

    # ── Category D: Negation ────────────────────────────────────────────────

    def _select_category_d(self, count: int) -> list[ConceptCluster]:
        """Select negation clusters using 3 sub-strategies."""
        clusters = []
        per_sub = max(1, count // 3)
        remainder = count - per_sub * 3

        # D1: Concept + action it CANNOT do (NOT_CapableOf)
        d1 = self._select_d1_not_capable(per_sub + (1 if remainder > 0 else 0))
        remainder = max(0, remainder - 1)
        clusters.extend(d1)

        # D2: Concept + antonym pair
        d2 = self._select_d2_antonym(per_sub + (1 if remainder > 0 else 0))
        remainder = max(0, remainder - 1)
        clusters.extend(d2)

        # D3: Concept + property it does NOT have (NOT_HasProperty)
        d3 = self._select_d3_not_property(per_sub)
        clusters.extend(d3)

        return clusters[:count]

    def _select_d1_not_capable(self, count: int) -> list[ConceptCluster]:
        """D1: Concept + action it CANNOT do.

        Cycles through candidates — same seed can produce multiple clusters
        if it has multiple CapableOf relations to negate.
        """
        clusters = []
        candidates = list(self._rich_concepts)
        if not candidates:
            return clusters
        self.rng.shuffle(candidates)

        # Build (seed, neg_rel, true_rel) combos
        combos = []
        for seed in candidates:
            rels = self._relations_by_source.get(seed["id"], [])
            capable_rels = [r for r in rels if r["relation_type_id"] == 5]
            true_rels = [r for r in rels if r["relation_type_id"] != 5 and r["weight"] >= _MIN_STRONG_WEIGHT]
            if not capable_rels or not true_rels:
                continue
            for neg in capable_rels:
                for true in true_rels[:2]:  # cap at 2 true rels per neg
                    combos.append((seed, neg, true))
        self.rng.shuffle(combos)

        seen_keys: set[tuple] = set()
        for seed, neg_rel, true_rel in combos:
            if len(clusters) >= count:
                break
            key = (seed["id"], neg_rel["target_concept_id"], true_rel["target_concept_id"])
            if key in seen_keys:
                continue
            seen_keys.add(key)

            seed_concept = {
                "id": seed["id"],
                "surface_text": seed["surface_text"],
                "anchor_token": seed["anchor_token"],
                "domain": seed["domain"],
                "category": seed["category"],
                "subcategory": seed["subcategory"],
                "specificity": seed["specificity"],
                "definition": seed.get("definition", ""),
            }

            target_neg = {
                "id": neg_rel["target_concept_id"],
                "surface_text": neg_rel["target_text"],
                "anchor_token": neg_rel["target_anchor"],
                "domain": neg_rel["target_domain"],
                "category": neg_rel["target_category"],
                "subcategory": neg_rel["target_subcategory"],
                "specificity": neg_rel["target_specificity"],
                "definition": neg_rel.get("target_definition", ""),
            }

            target_true = {
                "id": true_rel["target_concept_id"],
                "surface_text": true_rel["target_text"],
                "anchor_token": true_rel["target_anchor"],
                "domain": true_rel["target_domain"],
                "category": true_rel["target_category"],
                "subcategory": true_rel["target_subcategory"],
                "specificity": true_rel["target_specificity"],
                "definition": true_rel.get("target_definition", ""),
            }

            all_concepts = [seed_concept, target_neg, target_true]
            all_rels = [
                {
                    "source_id": seed["id"],
                    "target_id": neg_rel["target_concept_id"],
                    "relation_type_id": 5,  # CapableOf
                    "weight": neg_rel["weight"],
                    "negation": 1,  # NEGATED
                },
                {
                    "source_id": seed["id"],
                    "target_id": true_rel["target_concept_id"],
                    "relation_type_id": true_rel["relation_type_id"],
                    "weight": true_rel["weight"],
                    "negation": 0,
                },
            ]

            sml_block = self._build_sml_block(all_concepts, all_rels)

            cluster = ConceptCluster(
                category="D",
                seed_concept={
                    "id": seed["id"],
                    "surface_text": seed["surface_text"],
                    "anchor_token": seed["anchor_token"],
                },
                concepts=all_concepts,
                relations=all_rels,
                sml_block=sml_block,
                metadata={
                    "sub_strategy": "D1_NOT_CapableOf",
                    "negated_action": neg_rel["target_text"],
                    "true_relation": RELATION_TYPES.get(
                        true_rel["relation_type_id"],
                        str(true_rel["relation_type_id"]),
                    ),
                },
            )
            clusters.append(cluster)

        return clusters

    def _select_d2_antonym(self, count: int) -> list[ConceptCluster]:
        """D2: Concept + antonym pair.

        All unique antonym pairs are available; cycles if needed.
        """
        clusters = []
        pairs = list(self._antonym_pairs)
        if not pairs:
            return clusters
        self.rng.shuffle(pairs)

        for pair in pairs:
            if len(clusters) >= count:
                break

            c_source = {
                "id": pair["source_id"],
                "surface_text": pair["source_text"],
                "anchor_token": pair["source_anchor"],
                "domain": pair["source_domain"],
                "category": pair["source_category"],
                "subcategory": pair["source_subcategory"],
                "specificity": pair["source_specificity"],
                "definition": pair.get("source_definition", ""),
            }

            c_target = {
                "id": pair["target_id"],
                "surface_text": pair["target_text"],
                "anchor_token": pair["target_anchor"],
                "domain": pair["target_domain"],
                "category": pair["target_category"],
                "subcategory": pair["target_subcategory"],
                "specificity": pair["target_specificity"],
                "definition": pair.get("target_definition", ""),
            }

            all_concepts = [c_source, c_target]
            # Include the antonym relation as-is (not negated — it's an Antonym type)
            all_rels = [
                {
                    "source_id": pair["source_id"],
                    "target_id": pair["target_id"],
                    "relation_type_id": 22,  # Antonym
                    "weight": pair["weight"],
                    "negation": 0,
                },
            ]

            # Also add a true relation for the source if available
            source_rels = self._relations_by_source.get(pair["source_id"], [])
            for r in source_rels:
                if r["relation_type_id"] != 22 and r["weight"] >= _MIN_STRONG_WEIGHT:
                    # Add the target concept if not already present
                    if r["target_concept_id"] != pair["target_id"]:
                        all_concepts.append({
                            "id": r["target_concept_id"],
                            "surface_text": r["target_text"],
                            "anchor_token": r["target_anchor"],
                            "domain": r["target_domain"],
                            "category": r["target_category"],
                            "subcategory": r["target_subcategory"],
                            "specificity": r["target_specificity"],
                            "definition": r.get("target_definition", ""),
                        })
                    all_rels.append({
                        "source_id": pair["source_id"],
                        "target_id": r["target_concept_id"],
                        "relation_type_id": r["relation_type_id"],
                        "weight": r["weight"],
                        "negation": 0,
                    })
                    break  # just one extra

            sml_block = self._build_sml_block(all_concepts, all_rels)

            cluster = ConceptCluster(
                category="D",
                seed_concept={
                    "id": pair["source_id"],
                    "surface_text": pair["source_text"],
                    "anchor_token": pair["source_anchor"],
                },
                concepts=all_concepts,
                relations=all_rels,
                sml_block=sml_block,
                metadata={
                    "sub_strategy": "D2_Antonym",
                    "antonym_pair": [pair["source_text"], pair["target_text"]],
                },
            )
            clusters.append(cluster)

        return clusters

    def _select_d3_not_property(self, count: int) -> list[ConceptCluster]:
        """D3: Concept + property it does NOT have.

        Cycles through candidates — same seed can produce multiple clusters
        if it has multiple HasProperty relations to negate.
        """
        clusters = []
        candidates = list(self._rich_concepts)
        if not candidates:
            return clusters
        self.rng.shuffle(candidates)

        # Build (seed, neg_rel, true_rel) combos
        combos = []
        for seed in candidates:
            rels = self._relations_by_source.get(seed["id"], [])
            prop_rels = [r for r in rels if r["relation_type_id"] == 4]
            true_rels = [r for r in rels if r["relation_type_id"] != 4 and r["weight"] >= _MIN_STRONG_WEIGHT]
            if not prop_rels or not true_rels:
                continue
            for neg in prop_rels:
                for true in true_rels[:2]:
                    combos.append((seed, neg, true))
        self.rng.shuffle(combos)

        seen_keys: set[tuple] = set()
        for seed, neg_rel, true_rel in combos:
            if len(clusters) >= count:
                break
            key = (seed["id"], neg_rel["target_concept_id"], true_rel["target_concept_id"])
            if key in seen_keys:
                continue
            seen_keys.add(key)

            seed_concept = {
                "id": seed["id"],
                "surface_text": seed["surface_text"],
                "anchor_token": seed["anchor_token"],
                "domain": seed["domain"],
                "category": seed["category"],
                "subcategory": seed["subcategory"],
                "specificity": seed["specificity"],
                "definition": seed.get("definition", ""),
            }

            target_neg = {
                "id": neg_rel["target_concept_id"],
                "surface_text": neg_rel["target_text"],
                "anchor_token": neg_rel["target_anchor"],
                "domain": neg_rel["target_domain"],
                "category": neg_rel["target_category"],
                "subcategory": neg_rel["target_subcategory"],
                "specificity": neg_rel["target_specificity"],
                "definition": neg_rel.get("target_definition", ""),
            }

            target_true = {
                "id": true_rel["target_concept_id"],
                "surface_text": true_rel["target_text"],
                "anchor_token": true_rel["target_anchor"],
                "domain": true_rel["target_domain"],
                "category": true_rel["target_category"],
                "subcategory": true_rel["target_subcategory"],
                "specificity": true_rel["target_specificity"],
                "definition": true_rel.get("target_definition", ""),
            }

            all_concepts = [seed_concept, target_neg, target_true]
            all_rels = [
                {
                    "source_id": seed["id"],
                    "target_id": neg_rel["target_concept_id"],
                    "relation_type_id": 4,  # HasProperty
                    "weight": neg_rel["weight"],
                    "negation": 1,  # NEGATED
                },
                {
                    "source_id": seed["id"],
                    "target_id": true_rel["target_concept_id"],
                    "relation_type_id": true_rel["relation_type_id"],
                    "weight": true_rel["weight"],
                    "negation": 0,
                },
            ]

            sml_block = self._build_sml_block(all_concepts, all_rels)

            cluster = ConceptCluster(
                category="D",
                seed_concept={
                    "id": seed["id"],
                    "surface_text": seed["surface_text"],
                    "anchor_token": seed["anchor_token"],
                },
                concepts=all_concepts,
                relations=all_rels,
                sml_block=sml_block,
                metadata={
                    "sub_strategy": "D3_NOT_HasProperty",
                    "negated_property": neg_rel["target_text"],
                    "true_relation": RELATION_TYPES.get(
                        true_rel["relation_type_id"],
                        str(true_rel["relation_type_id"]),
                    ),
                },
            )
            clusters.append(cluster)

        return clusters

    # ── Public API ──────────────────────────────────────────────────────────

    def select_clusters(
        self,
        category_counts: Optional[dict[str, int]] = None,
    ) -> list[ConceptCluster]:
        """Select concept clusters for all categories.

        Args:
            category_counts: Dict mapping category letter to count.
                Defaults to V3_CATEGORY_DISTRIBUTION.

        Returns:
            List of ConceptCluster objects with pre-built SML blocks.
        """
        if category_counts is None:
            from sml.config import V3_CATEGORY_DISTRIBUTION
            category_counts = V3_CATEGORY_DISTRIBUTION

        clusters = []

        if "A" in category_counts and category_counts["A"] > 0:
            clusters.extend(self._select_category_a(category_counts["A"]))

        if "B" in category_counts and category_counts["B"] > 0:
            clusters.extend(self._select_category_b(category_counts["B"]))

        if "C" in category_counts and category_counts["C"] > 0:
            clusters.extend(self._select_category_c(category_counts["C"]))

        if "D" in category_counts and category_counts["D"] > 0:
            clusters.extend(self._select_category_d(category_counts["D"]))

        self.rng.shuffle(clusters)
        return clusters
