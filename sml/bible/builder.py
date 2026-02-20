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
