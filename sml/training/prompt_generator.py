"""Template-based prompt generator for training data at scale.

Generates diverse training prompts by sampling concept/relation pairs from the
Bible DB and filling question templates. Each prompt references known concepts
and relations in the encoder, producing rich SML blocks.
"""

import random
import sqlite3
from collections import defaultdict

# ── Relation-based template banks ──────────────────────────────────────────────
# Keys are relation_type_id. Templates use {source} and {target} placeholders
# filled from real Bible relation pairs.

_RELATION_TEMPLATES = {
    4: [  # HasProperty: source=entity, target=property
        "Is {source} {target}?",
        "Would you describe {source} as {target}?",
        "How {target} is {source}?",
        "Can {source} be described as {target}?",
        "Is it true that {source} is {target}?",
        "Tell me, is {source} {target}?",
        "What is {source} like? Is it {target}?",
        "Describe {source}. Is it {target}?",
        "Would you say {source} is {target}?",
        "Is {source} generally considered {target}?",
    ],
    5: [  # CapableOf: source=entity, target=action
        "Can {source} {target}?",
        "Is {source} able to {target}?",
        "What happens when {source} tries to {target}?",
        "Does {source} {target}?",
        "Can a {source} {target}?",
        "Tell me, can {source} {target}?",
        "Is it true that {source} can {target}?",
        "Do you think {source} can {target}?",
        "What do you know about whether {source} can {target}?",
        "Is {source} known to {target}?",
    ],
    1: [  # IsA: source=entity, target=category
        "Is {source} a type of {target}?",
        "What kind of {target} is {source}?",
        "Would you classify {source} as a {target}?",
        "Is {source} a {target}?",
        "Can {source} be called a {target}?",
        "Does {source} belong to the category of {target}?",
        "Is it correct to say that {source} is a {target}?",
        "Tell me about {source}. Is it a {target}?",
        "Would you say {source} is a kind of {target}?",
        "Is {source} considered a {target}?",
    ],
    6: [  # AtLocation: source=entity, target=location
        "Where can you find {source}?",
        "Is {source} found at {target}?",
        "What can you find at {target}?",
        "Can you find {source} at {target}?",
        "Is {source} located at {target}?",
        "Where is {source} usually found?",
        "What things are at {target}?",
        "Where does {source} usually go?",
        "Is {target} a place where you might find {source}?",
        "Do you expect to find {source} at {target}?",
    ],
    7: [  # Causes
        "What does {source} cause?",
        "Does {source} lead to {target}?",
        "What happens because of {source}?",
        "Can {source} cause {target}?",
        "What is the result of {source}?",
        "Does {source} result in {target}?",
        "What effect does {source} have?",
        "Is {target} caused by {source}?",
    ],
    12: [  # UsedFor
        "What is {source} used for?",
        "Is {source} used for {target}?",
        "What can you do with {source}?",
        "Can you use {source} for {target}?",
        "What purpose does {source} serve?",
        "Is {source} useful for {target}?",
        "Why do people use {source}?",
        "What do you use {source} for?",
        "What is the purpose of {source}?",
        "Is {source} helpful for {target}?",
    ],
    16: [  # MadeOf
        "What is {source} made of?",
        "Is {source} made from {target}?",
        "What material is in {source}?",
        "Does {source} contain {target}?",
        "What is {source} composed of?",
        "Is {source} made with {target}?",
        "What goes into making {source}?",
        "Is {target} a component of {source}?",
    ],
    22: [  # Antonym
        "What is the opposite of {source}?",
        "Are {source} and {target} opposites?",
        "Is {target} the opposite of {source}?",
        "If something is not {source}, is it {target}?",
        "What is the antonym of {source}?",
        "How do {source} and {target} relate to each other?",
        "Is {source} the opposite of {target}?",
        "What word means the opposite of {source}?",
    ],
    3: [  # HasA
        "Does {source} have {target}?",
        "What does {source} have?",
        "Does {source} have a {target}?",
        "What parts does {source} have?",
        "Is {target} a part of {source}?",
        "Tell me what {source} has.",
        "Does a {source} possess a {target}?",
        "What body parts does {source} have?",
    ],
    18: [  # Desires
        "What does {source} want?",
        "Does {source} want {target}?",
        "What does {source} desire?",
        "Does {source} like {target}?",
        "What does {source} enjoy?",
        "Is {target} something {source} wants?",
        "What does {source} prefer?",
        "Does {source} crave {target}?",
    ],
}

# ── Property-specific templates (by property category) ─────────────────────────
# {source} is the entity that has the property.

_PROPERTY_QUERY_TEMPLATES = {
    1: [  # Color
        "What color is {source}?",
        "Describe the color of {source}.",
        "What color can {source} be?",
        "What colors are associated with {source}?",
    ],
    2: [  # Size
        "How big is {source}?",
        "What size is {source}?",
        "Is {source} large or small?",
        "Describe the size of {source}.",
    ],
    3: [  # Temperature
        "Is {source} hot or cold?",
        "What temperature is {source}?",
        "How hot is {source}?",
        "Describe the temperature of {source}.",
    ],
    4: [  # Speed
        "How fast is {source}?",
        "Is {source} fast or slow?",
        "What speed does {source} move at?",
    ],
    5: [  # Age
        "Is {source} old or new?",
        "How old is {source}?",
    ],
    6: [  # Luminosity
        "Is {source} bright or dark?",
        "How bright is {source}?",
        "Describe the brightness of {source}.",
    ],
    7: [  # Weight
        "Is {source} heavy or light?",
        "How heavy is {source}?",
        "What does {source} weigh?",
    ],
    8: [  # Texture
        "Is {source} soft or hard?",
        "What texture is {source}?",
        "How does {source} feel to the touch?",
    ],
    9: [  # Taste
        "What does {source} taste like?",
        "Is {source} sweet or salty?",
        "Describe the taste of {source}.",
    ],
    10: [  # Shape
        "What shape is {source}?",
        "Is {source} round?",
        "Describe the shape of {source}.",
    ],
    11: [  # Sound
        "Is {source} loud or quiet?",
        "What sound does {source} make?",
        "How noisy is {source}?",
    ],
    12: [  # Clarity
        "Is {source} clear?",
        "Can you see through {source}?",
        "How transparent is {source}?",
    ],
    13: [  # Personality
        "What personality does {source} have?",
        "How would you describe the character of {source}?",
        "Is {source} friendly?",
    ],
}

# ── Scene / compositional templates ────────────────────────────────────────────

_SCENE_TEMPLATES = [
    "The {adj} {entity} is at the {location}. Describe the scene.",
    "The {adj} {entity} is at the {location}. What is the {entity} doing?",
    "A {adj} {entity} is near the {location}. What do you see?",
    "The {adj} {entity} is at the {location}. What is nearby?",
    "Imagine a {adj} {entity} at the {location}. Describe what you see.",
    "There is a {adj} {entity} at the {location}. What happens next?",
    "A {adj} {entity} appeared at the {location}. Describe the situation.",
    "Picture a {adj} {entity} at the {location}. What is happening?",
]

_SCENE_TWO_ENTITY_TEMPLATES = [
    "{entity1} and {entity2} are at the {location}. What might happen?",
    "There is a {adj1} {entity1} and a {adj2} {entity2}. Compare them.",
    "A {adj1} {entity1} and a {adj2} {entity2} are at the {location}. Describe the scene.",
    "The {adj1} {entity1} met the {adj2} {entity2} at the {location}. What happens?",
]

# ── Negation templates ─────────────────────────────────────────────────────────

_NEGATION_CAPABLE_TEMPLATES = [
    "Can {entity} {action}?",
    "Is {entity} able to {action}?",
    "Does {entity} {action}?",
    "Can a {entity} {action}?",
    "Is it possible for {entity} to {action}?",
]

_NEGATION_PROPERTY_TEMPLATES = [
    "Is {entity} {property}?",
    "Would you describe {entity} as {property}?",
    "Can {entity} be described as {property}?",
    "Is it true that {entity} is {property}?",
]

# ── Comparison templates ───────────────────────────────────────────────────────

_COMPARISON_TEMPLATES = [
    "Is {entity1} bigger than {entity2}?",
    "Which is faster, {entity1} or {entity2}?",
    "How are {entity1} and {entity2} different?",
    "What do {entity1} and {entity2} have in common?",
    "Compare {entity1} and {entity2}.",
    "Which is heavier, {entity1} or {entity2}?",
    "How are {entity1} and {entity2} similar?",
    "What are the differences between {entity1} and {entity2}?",
]

# ── Entity-only / open-ended templates ─────────────────────────────────────────

_ENTITY_TEMPLATES = [
    "What is {entity}?",
    "Describe {entity}.",
    "Tell me about {entity}.",
    "What are the properties of {entity}?",
    "What do you know about {entity}?",
    "What can you tell me about {entity}?",
    "Explain what {entity} is.",
    "What is special about {entity}?",
]

# ── Target distribution for generated prompts ──────────────────────────────────

_TARGET_DISTRIBUTION = [
    ("has_property_direct", 0.15),
    ("has_property_category", 0.10),
    ("capable_of", 0.15),
    ("at_location", 0.12),
    ("is_a", 0.08),
    ("causes", 0.07),
    ("used_for", 0.07),
    ("scene", 0.08),
    ("negation", 0.07),
    ("comparison", 0.04),
    ("minor_relations", 0.04),
    ("entity_only", 0.03),
]


class PromptGenerator:
    """Generate diverse training prompts by sampling concept/relation pairs from the Bible DB."""

    def __init__(self, bible_path: str, seed: int = 42):
        self.rng = random.Random(seed)
        self.conn = sqlite3.connect(bible_path)
        self._load_data()

    def _load_data(self):
        """Load concepts and relations from Bible, grouped for template filling."""
        cur = self.conn.cursor()

        # Load all concepts: id -> {surface_text, domain, category, subcategory}
        self.concepts = {}
        for row in cur.execute(
            "SELECT id, surface_text, domain, category, subcategory FROM concepts"
        ):
            self.concepts[row[0]] = {
                "surface_text": row[1],
                "domain": row[2],
                "category": row[3],
                "subcategory": row[4],
            }

        # Load all relations grouped by type: type_id -> [(source_text, target_text, weight)]
        self.relations_by_type = defaultdict(list)
        for row in cur.execute(
            "SELECT r.relation_type_id, c1.surface_text, c2.surface_text, r.weight "
            "FROM relations r "
            "JOIN concepts c1 ON r.source_id = c1.id "
            "JOIN concepts c2 ON r.target_id = c2.id"
        ):
            self.relations_by_type[row[0]].append((row[1], row[2], row[3]))

        # Entities usable as sentence subjects (domain 1=physical or 2=abstract)
        self.entity_texts = [
            c["surface_text"] for c in self.concepts.values()
            if c["domain"] in (1, 2)
        ]

        # All action texts (domain 3)
        self.all_action_texts = [
            c["surface_text"] for c in self.concepts.values()
            if c["domain"] == 3
        ]

        # All property texts (domain 4)
        self.all_property_texts = [
            c["surface_text"] for c in self.concepts.values()
            if c["domain"] == 4
        ]

        # Location texts (domain 1, category 4 = places)
        self.location_texts = [
            c["surface_text"] for c in self.concepts.values()
            if c["domain"] == 1 and c["category"] == 4
        ]

        # Property category lookup: property_text -> category
        self.property_category = {}
        for c in self.concepts.values():
            if c["domain"] == 4:
                self.property_category[c["surface_text"]] = c["category"]

        # Build per-entity indexes from relations
        self.has_property_by_entity = defaultdict(list)
        self.at_location_by_entity = defaultdict(list)
        self.capable_of_set = set()
        self.has_property_set = set()

        for src, tgt, _ in self.relations_by_type.get(4, []):
            self.has_property_by_entity[src].append(tgt)
            self.has_property_set.add((src, tgt))

        for src, tgt, _ in self.relations_by_type.get(6, []):
            self.at_location_by_entity[src].append(tgt)

        for src, tgt, _ in self.relations_by_type.get(5, []):
            self.capable_of_set.add((src, tgt))

        # Entities that participate in key relation types (as source)
        self.entities_with_capabilities = list({
            src for src, _, _ in self.relations_by_type.get(5, [])
        })
        self.entities_with_properties = list({
            src for src, _, _ in self.relations_by_type.get(4, [])
        })

    # ── Pool generators ────────────────────────────────────────────────────

    def _gen_relation_prompts(self, relation_type_id: int) -> list[str]:
        """Generate all possible prompts for a relation type from its template bank."""
        templates = _RELATION_TEMPLATES.get(relation_type_id, [])
        if not templates:
            return []
        pairs = self.relations_by_type.get(relation_type_id, [])
        prompts = set()
        for source_text, target_text, _ in pairs:
            for template in templates:
                prompts.add(template.format(source=source_text, target=target_text))
        return list(prompts)

    def _gen_property_category_prompts(self) -> list[str]:
        """Generate property-specific prompts using category-aware templates."""
        prompts = set()
        for entity_text, prop_text in self.has_property_set:
            cat = self.property_category.get(prop_text, 0)
            for template in _PROPERTY_QUERY_TEMPLATES.get(cat, []):
                prompts.add(template.format(source=entity_text))
        return list(prompts)

    def _gen_scene_prompts(self, max_prompts: int = 5000) -> list[str]:
        """Generate scene/compositional prompts by sampling entity + property + location."""
        prompts = set()

        entities = list(self.entities_with_properties)
        locs_fallback = self.location_texts[:10] if self.location_texts else ["a room"]

        # Single entity scenes: sample combinations instead of full cross-product
        self.rng.shuffle(entities)
        for entity in entities:
            if len(prompts) >= max_prompts:
                break
            props = self.has_property_by_entity.get(entity, [])
            locs = self.at_location_by_entity.get(entity, []) or locs_fallback
            prop = self.rng.choice(props) if props else None
            loc = self.rng.choice(locs)
            if prop:
                tpl = self.rng.choice(_SCENE_TEMPLATES)
                prompts.add(tpl.format(adj=prop, entity=entity, location=loc))

        # Two entity scenes: sample pairs instead of full cross-product
        paired = [
            e for e in self.entities_with_properties
            if self.at_location_by_entity.get(e)
        ]
        if len(paired) >= 2:
            attempts = min(max_prompts - len(prompts), len(paired) * 2)
            for _ in range(max(0, attempts)):
                e1, e2 = self.rng.sample(paired, 2)
                a1 = self.rng.choice(self.has_property_by_entity[e1])
                a2 = self.rng.choice(self.has_property_by_entity[e2])
                loc = self.rng.choice(self.at_location_by_entity[e1])
                tpl = self.rng.choice(_SCENE_TWO_ENTITY_TEMPLATES)
                prompts.add(tpl.format(
                    entity1=e1, entity2=e2,
                    adj1=a1, adj2=a2, location=loc,
                ))

        return list(prompts)

    def _gen_negation_prompts(self, max_prompts: int = 5000) -> list[str]:
        """Generate negation prompts by sampling entity/action and entity/property mismatches."""
        prompts = set()

        # CapableOf negation: sample entity + action pairs where entity CANNOT do the action
        if self.entities_with_capabilities and self.all_action_texts:
            attempts = max_prompts * 2  # oversample to account for positive hits
            for _ in range(attempts):
                if len(prompts) >= max_prompts // 2:
                    break
                entity = self.rng.choice(self.entities_with_capabilities)
                action = self.rng.choice(self.all_action_texts)
                if (entity, action) not in self.capable_of_set:
                    tpl = self.rng.choice(_NEGATION_CAPABLE_TEMPLATES)
                    prompts.add(tpl.format(entity=entity, action=action))

        # HasProperty negation: sample entity + property pairs where entity does NOT have it
        if self.entities_with_properties and self.all_property_texts:
            attempts = max_prompts * 2
            for _ in range(attempts):
                if len(prompts) >= max_prompts:
                    break
                entity = self.rng.choice(self.entities_with_properties)
                prop = self.rng.choice(self.all_property_texts)
                if (entity, prop) not in self.has_property_set:
                    tpl = self.rng.choice(_NEGATION_PROPERTY_TEMPLATES)
                    prompts.add(tpl.format(entity=entity, property=prop))

        return list(prompts)

    def _gen_comparison_prompts(self, max_prompts: int = 5000) -> list[str]:
        """Generate comparison prompts by sampling same-category entity pairs."""
        prompts = set()

        # Group entities by (domain, category) for meaningful comparisons
        groups = defaultdict(list)
        for c in self.concepts.values():
            if c["domain"] in (1, 2):
                groups[(c["domain"], c["category"])].append(c["surface_text"])

        # Only keep groups with at least 2 entities
        valid_groups = [ents for ents in groups.values() if len(ents) >= 2]
        if not valid_groups:
            return []

        attempts = max_prompts * 2
        for _ in range(attempts):
            if len(prompts) >= max_prompts:
                break
            group = self.rng.choice(valid_groups)
            e1, e2 = self.rng.sample(group, 2)
            tpl = self.rng.choice(_COMPARISON_TEMPLATES)
            prompts.add(tpl.format(entity1=e1, entity2=e2))

        return list(prompts)

    def _gen_entity_prompts(self, max_prompts: int = 5000) -> list[str]:
        """Generate entity-only / open-ended prompts by sampling entities."""
        prompts = set()
        entities = list(self.entity_texts)
        self.rng.shuffle(entities)
        for entity in entities:
            if len(prompts) >= max_prompts:
                break
            tpl = self.rng.choice(_ENTITY_TEMPLATES)
            prompts.add(tpl.format(entity=entity))
        return list(prompts)

    # ── Main generation method ─────────────────────────────────────────────

    def generate(self, num_prompts: int) -> list[str]:
        """Generate num_prompts unique prompts. Always includes MICRO_PROMPTS first."""
        from sml.training.data_generator import MICRO_PROMPTS

        base = list(MICRO_PROMPTS)
        if num_prompts <= len(base):
            return base[:num_prompts]

        remaining = num_prompts - len(base)

        # Per-pool cap: 3x the target allocation (enough for dedup + overflow)
        pool_cap = max(remaining * 3, 15000)

        # Build prompt pools for each category (capped to avoid OOM)
        pools = {
            "has_property_direct": self._gen_relation_prompts(4),
            "has_property_category": self._gen_property_category_prompts(),
            "capable_of": self._gen_relation_prompts(5),
            "at_location": self._gen_relation_prompts(6),
            "is_a": self._gen_relation_prompts(1),
            "causes": self._gen_relation_prompts(7),
            "used_for": self._gen_relation_prompts(12),
            "scene": self._gen_scene_prompts(max_prompts=pool_cap),
            "negation": self._gen_negation_prompts(max_prompts=pool_cap),
            "comparison": self._gen_comparison_prompts(max_prompts=pool_cap),
            "minor_relations": (
                self._gen_relation_prompts(16)
                + self._gen_relation_prompts(3)
                + self._gen_relation_prompts(18)
                + self._gen_relation_prompts(22)
            ),
            "entity_only": self._gen_entity_prompts(max_prompts=pool_cap),
        }

        # Deduplicate each pool and exclude MICRO_PROMPTS
        micro_lower = {p.lower() for p in base}
        for key in pools:
            seen = set()
            deduped = []
            for p in pools[key]:
                low = p.lower()
                if low not in seen and low not in micro_lower:
                    seen.add(low)
                    deduped.append(p)
            pools[key] = deduped

        # Calculate per-category targets
        targets = {}
        total_assigned = 0
        for name, pct in _TARGET_DISTRIBUTION:
            t = max(1, int(pct * remaining))
            targets[name] = t
            total_assigned += t

        # Fix rounding — distribute remainder across categories
        diff = remaining - total_assigned
        keys = [name for name, _ in _TARGET_DISTRIBUTION]
        for i in range(max(0, diff)):
            targets[keys[i % len(keys)]] += 1

        # Sample from each pool, collecting overflow for redistribution
        sampled = []
        overflow = []
        for name, _ in _TARGET_DISTRIBUTION:
            target = targets[name]
            pool = pools[name]
            self.rng.shuffle(pool)
            sampled.extend(pool[:target])
            overflow.extend(pool[target:])

        # Fill any shortfall from overflow
        if len(sampled) < remaining and overflow:
            self.rng.shuffle(overflow)
            sampled.extend(overflow[:remaining - len(sampled)])

        # Final deduplication
        seen = set(micro_lower)
        unique = []
        for p in sampled:
            low = p.lower()
            if low not in seen:
                seen.add(low)
                unique.append(p)

        # If still short after dedup, cycle from unique generated prompts
        if unique and len(unique) < remaining:
            full = list(unique)
            idx = 0
            while len(full) < remaining:
                full.append(unique[idx % len(unique)])
                idx += 1
            unique = full

        self.rng.shuffle(unique)
        return base + unique[:remaining]

    def close(self):
        """Close the database connection."""
        self.conn.close()
