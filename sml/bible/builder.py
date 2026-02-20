"""Full SML Bible builder from ConceptNet 5.7 + WordNet."""
import csv
import gzip
import json
import sqlite3
import urllib.request
from pathlib import Path
from typing import Optional

from sml.bible.schema import create_bible_db
from sml.config import CONCEPTNET_MIN_WEIGHT, CONCEPTNET_URL


def _parse_conceptnet_uri(uri: str) -> Optional[str]:
    """Extract English surface text from a ConceptNet URI like /c/en/dog."""
    parts = uri.split("/")
    if len(parts) >= 4 and parts[2] == "en":
        return parts[3].replace("_", " ")
    return None


def _download_conceptnet(cache_path: Path) -> Path:
    """Download ConceptNet CSV dump if not already cached."""
    if cache_path.exists():
        print(f"Using cached ConceptNet dump: {cache_path}")
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ConceptNet from {CONCEPTNET_URL}...")
    print("This is a large file (~500MB) and may take a while.")
    urllib.request.urlretrieve(CONCEPTNET_URL, str(cache_path))
    print(f"Downloaded to {cache_path}")
    return cache_path


def _get_wordnet_taxonomy(surface_text: str) -> dict:
    """Get WordNet taxonomy info for a surface text.

    Returns dict with domain, category, subcategory, specificity fields.
    Uses NLTK WordNet to walk hypernym chains.
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return {"domain": 0, "category": 0, "subcategory": 0, "specificity": 0}

    synsets = wn.synsets(surface_text.replace(" ", "_"))
    if not synsets:
        return {"domain": 0, "category": 0, "subcategory": 0, "specificity": 0}

    synset = synsets[0]  # Take first (most common) sense
    pos = synset.pos()

    # Domain mapping based on POS and hypernym chain
    domain = 1  # default: physical
    if pos == "v":
        domain = 3  # action
    elif pos in ("a", "s", "r"):
        domain = 4  # property (adjective/adverb)

    # Walk hypernym chain to get category info
    hypernyms = synset.hypernym_paths()
    if not hypernyms:
        return {"domain": domain, "category": 0, "subcategory": 0, "specificity": 0}

    chain = hypernyms[0]  # Take first hypernym path
    category = 0
    subcategory = 0
    specificity = 0

    # Map high-level hypernyms to category IDs
    top_level_map = {
        "entity.n.01": 0,
        "physical_entity.n.01": 1,
        "abstraction.n.06": 2,
        "thing.n.12": 1,
        "object.n.01": 2,
        "whole.n.02": 2,
        "living_thing.n.01": 1,
        "organism.n.01": 1,
        "animal.n.01": 2,
        "plant.n.02": 3,
        "person.n.01": 1,
        "artifact.n.01": 2,
        "food.n.01": 3,
        "substance.n.01": 3,
        "location.n.01": 4,
    }

    for ancestor in chain:
        name = ancestor.name()
        if name in top_level_map:
            if category == 0:
                category = top_level_map[name]
            elif subcategory == 0:
                subcategory = top_level_map[name]

    # Specificity: depth in taxonomy (capped at 5)
    specificity = min(len(chain) - 1, 5) if len(chain) > 1 else 0

    return {
        "domain": domain,
        "category": category,
        "subcategory": subcategory,
        "specificity": specificity,
    }


# Curated mapping: surface_text -> property category ID
_PROPERTY_CATEGORY_SEEDS = {
    # 1: color
    "red": 1, "blue": 1, "green": 1, "yellow": 1, "brown": 1,
    "white": 1, "black": 1, "orange": 1, "purple": 1, "pink": 1,
    "gray": 1, "grey": 1, "golden": 1, "silver": 1, "crimson": 1,
    "scarlet": 1, "violet": 1, "maroon": 1, "turquoise": 1, "beige": 1,
    "tan": 1, "ivory": 1, "indigo": 1, "magenta": 1, "teal": 1,
    "colorful": 1, "colourful": 1, "multicolored": 1,
    # 2: size
    "big": 2, "small": 2, "tall": 2, "deep": 2, "large": 2,
    "tiny": 2, "huge": 2, "short": 2, "narrow": 2, "wide": 2,
    "vast": 2, "little": 2, "enormous": 2, "giant": 2, "massive": 2,
    "miniature": 2, "petite": 2, "compact": 2, "immense": 2, "thick": 2, "thin": 2,
    # 3: temperature
    "hot": 3, "cold": 3, "warm": 3, "cool": 3, "freezing": 3,
    "boiling": 3, "chilly": 3, "icy": 3, "lukewarm": 3, "tepid": 3, "scorching": 3,
    # 4: speed
    "fast": 4, "slow": 4, "quick": 4, "rapid": 4, "swift": 4, "speedy": 4, "sluggish": 4,
    # 5: age
    "old": 5, "new": 5, "young": 5, "ancient": 5, "modern": 5,
    "fresh": 5, "elderly": 5, "youthful": 5, "aged": 5, "juvenile": 5,
    # 6: luminosity
    "bright": 6, "dark": 6, "dim": 6, "shiny": 6, "dull": 6,
    "glowing": 6, "radiant": 6, "luminous": 6, "brilliant": 6, "faint": 6,
    # 7: weight
    "heavy": 7, "light": 7, "lightweight": 7, "weightless": 7,
    # 8: texture
    "soft": 8, "hard": 8, "rough": 8, "smooth": 8, "fuzzy": 8,
    "coarse": 8, "silky": 8, "bumpy": 8, "fluffy": 8, "rigid": 8, "tender": 8, "crispy": 8,
    # 9: taste
    "sweet": 9, "salty": 9, "bitter": 9, "sour": 9, "spicy": 9,
    "bland": 9, "savory": 9, "tangy": 9, "delicious": 9, "tasty": 9,
    # 10: shape
    "round": 10, "long": 10, "flat": 10, "curved": 10, "square": 10,
    "circular": 10, "oval": 10, "straight": 10, "triangular": 10, "cylindrical": 10, "spherical": 10,
    # 11: sound
    "quiet": 11, "loud": 11, "noisy": 11, "silent": 11, "deafening": 11, "mute": 11,
    # 12: clarity
    "clear": 12, "transparent": 12, "opaque": 12, "murky": 12, "foggy": 12, "cloudy": 12,
    # 13: personality/character
    "friendly": 13, "loyal": 13, "independent": 13, "cute": 13, "brave": 13,
    "gentle": 13, "shy": 13, "kind": 13, "mean": 13, "cruel": 13,
    "honest": 13, "generous": 13, "stubborn": 13, "playful": 13, "aggressive": 13,
    "calm": 13, "nervous": 13, "lazy": 13, "curious": 13, "clever": 13,
}

# Definition keyword -> category (for WordNet fallback)
_DEFINITION_HINTS = {
    1: ["color", "colour", "hue", "pigment", "chromatic"],
    2: ["size", "extent", "dimension", "stature", "height"],
    3: ["temperature", "thermal", "heat", "warmth", "cold"],
    4: ["speed", "velocity", "pace", "motion"],
    6: ["light", "luminous", "bright", "illuminat"],
    7: ["weight", "mass", "heaviness"],
    8: ["texture", "surface", "touch", "tactile"],
    9: ["taste", "flavor", "palate"],
    10: ["shape", "form", "geometry", "contour"],
    11: ["sound", "noise", "auditory", "acoustic"],
    12: ["clarity", "transparent", "visibility", "optic"],
}


def _classify_property_category(surface_text: str) -> int:
    """Classify a property-domain concept into a specific category.

    Returns category ID (1-13) or 0 for unclassified.
    """
    # Tier 1: curated seed lookup
    cat = _PROPERTY_CATEGORY_SEEDS.get(surface_text.lower())
    if cat is not None:
        return cat

    # Tier 2: WordNet definition heuristic
    try:
        from nltk.corpus import wordnet as wn

        synsets = wn.synsets(surface_text.replace(" ", "_"))
        if synsets:
            defn = synsets[0].definition().lower()
            for cat_id, keywords in _DEFINITION_HINTS.items():
                if any(kw in defn for kw in keywords):
                    return cat_id
    except Exception:
        pass

    return 0


def _generate_anchor_token(surface_text: str, concept_id: int) -> str:
    """Generate string-anchored token like dog_1001."""
    clean = surface_text.lower().replace(" ", "_").replace("-", "_")
    # Keep only alphanumeric and underscores
    clean = "".join(c for c in clean if c.isalnum() or c == "_")
    return f"{clean}_{concept_id}"


def build_full_bible(
    db_path: str,
    conceptnet_cache: Optional[str] = None,
    max_concepts: int = 100000,
    progress_interval: int = 10000,
) -> None:
    """Build the full SML Bible from ConceptNet and WordNet.

    Args:
        db_path: Path for the output SQLite database.
        conceptnet_cache: Path to cached ConceptNet CSV.gz file.
            If None, downloads to data/conceptnet-assertions-5.7.0.csv.gz.
        max_concepts: Maximum number of concepts to import.
        progress_interval: Print progress every N concepts.
    """
    from tqdm import tqdm

    # Setup paths
    db_path = str(db_path)
    if conceptnet_cache is None:
        conceptnet_cache = str(Path(db_path).parent / "conceptnet-assertions-5.7.0.csv.gz")

    # Download ConceptNet if needed
    cn_path = _download_conceptnet(Path(conceptnet_cache))

    # Create fresh database
    create_bible_db(db_path)
    conn = sqlite3.connect(db_path)

    # Phase 1: Extract English concepts and relations from ConceptNet
    print("Phase 1: Parsing ConceptNet assertions...")
    concepts = {}  # surface_text -> {uri, id, ...}
    raw_relations = []  # [(source_text, target_text, relation, weight)]

    # Map ConceptNet relation URIs to our IDs
    rel_uri_to_id = {
        "/r/IsA": 1, "/r/PartOf": 2, "/r/HasA": 3, "/r/HasProperty": 4,
        "/r/CapableOf": 5, "/r/AtLocation": 6, "/r/Causes": 7,
        "/r/HasPrerequisite": 8, "/r/HasFirstSubevent": 9,
        "/r/HasLastSubevent": 10, "/r/MotivatedByGoal": 11, "/r/UsedFor": 12,
        "/r/CreatedBy": 13, "/r/DefinedAs": 14, "/r/SymbolOf": 15,
        "/r/MadeOf": 16, "/r/ReceivesAction": 17, "/r/Desires": 18,
        "/r/CausesDesire": 19, "/r/HasContext": 20, "/r/SimilarTo": 21,
        "/r/Antonym": 22, "/r/DerivedFrom": 23, "/r/RelatedTo": 24,
        "/r/FormOf": 25, "/r/EtymologicallyRelatedTo": 26, "/r/Synonym": 27,
        "/r/MannerOf": 28, "/r/LocatedNear": 29,
        "/r/dbpedia/genre": 31, "/r/dbpedia/occupation": 32,
        "/r/dbpedia/language": 33, "/r/dbpedia/capital": 34,
    }

    concept_id_counter = 1

    with gzip.open(str(cn_path), "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc="Reading ConceptNet", unit=" rows"):
            if len(row) < 5:
                continue

            rel_uri = row[1]
            source_uri = row[2]
            target_uri = row[3]

            # Parse weight from the JSON metadata column
            try:
                meta = json.loads(row[4])
                weight = meta.get("weight", 0.0)
            except (json.JSONDecodeError, IndexError):
                weight = 0.0

            # Filter: English only, minimum weight
            if weight < CONCEPTNET_MIN_WEIGHT:
                continue

            source_text = _parse_conceptnet_uri(source_uri)
            target_text = _parse_conceptnet_uri(target_uri)

            if source_text is None or target_text is None:
                continue

            rel_id = rel_uri_to_id.get(rel_uri)
            if rel_id is None:
                continue

            # Track unique concepts
            for text, uri in [(source_text, source_uri), (target_text, target_uri)]:
                if text not in concepts and len(concepts) < max_concepts:
                    concepts[text] = {
                        "uri": uri,
                        "id": concept_id_counter,
                        "surface_text": text,
                    }
                    concept_id_counter += 1

            # Store relation for later
            if source_text in concepts and target_text in concepts:
                raw_relations.append((source_text, target_text, rel_id, min(weight / 10.0, 1.0)))

    print(f"Extracted {len(concepts)} unique concepts and {len(raw_relations)} relations")

    # Phase 2: Enrich with WordNet taxonomy
    print("Phase 2: Enriching with WordNet taxonomy...")
    concept_rows = []
    for i, (text, info) in enumerate(concepts.items()):
        if i > 0 and i % progress_interval == 0:
            print(f"  Processed {i}/{len(concepts)} concepts...")

        taxonomy = _get_wordnet_taxonomy(text)

        # Reclassify property categories to match encoder scheme
        if taxonomy["domain"] == 4:
            taxonomy["category"] = _classify_property_category(text)

        anchor = _generate_anchor_token(text, info["id"])

        # Try to get WordNet definition
        definition = ""
        try:
            from nltk.corpus import wordnet as wn
            synsets = wn.synsets(text.replace(" ", "_"))
            if synsets:
                defn = synsets[0].definition()
                if defn:
                    definition = defn
        except Exception:
            pass

        concept_rows.append((
            info["id"],
            info["uri"],
            text,
            anchor,
            taxonomy["domain"],
            taxonomy["category"],
            taxonomy["subcategory"],
            taxonomy["specificity"],
            definition,
        ))

    # Phase 3: Insert into database
    print("Phase 3: Inserting concepts into database...")
    conn.executemany(
        """INSERT OR IGNORE INTO concepts
           (id, uri, surface_text, anchor_token, domain, category, subcategory, specificity, definition)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        concept_rows,
    )

    print("Phase 4: Inserting relations...")
    relation_rows = []
    for source_text, target_text, rel_id, weight in raw_relations:
        if source_text in concepts and target_text in concepts:
            relation_rows.append((
                concepts[source_text]["id"],
                concepts[target_text]["id"],
                rel_id,
                weight,
            ))

    conn.executemany(
        "INSERT INTO relations (source_id, target_id, relation_type_id, weight) VALUES (?, ?, ?, ?)",
        relation_rows,
    )

    conn.commit()

    # Rebuild FTS index
    print("Phase 5: Rebuilding FTS index...")
    conn.execute("INSERT INTO concepts_fts(concepts_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()

    print(f"\nFull Bible built: {len(concepts)} concepts, {len(relation_rows)} relations at {db_path}")
