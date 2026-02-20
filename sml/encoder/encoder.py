"""Rule-based SML Encoder — converts text to SML arrays via spaCy + Bible."""
from typing import Optional

from sml.bible.query import Bible
from sml.encoder.formatter import format_sml_block


class SMLEncoder:
    """Encodes natural language text into SML blocks using spaCy and the Bible."""

    def __init__(self, bible_path: str, spacy_model: str = "en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(spacy_model)
        self.bible = Bible(bible_path)
        self._embedder = None  # Lazy-loaded sentence-transformers model

    def close(self):
        self.bible.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    @property
    def embedder(self):
        """Lazy-load sentence-transformers for disambiguation."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                pass
        return self._embedder

    def _resolve_concept(self, token_text: str, context: str = "") -> Optional[dict]:
        """Look up a token in the Bible, with fuzzy fallback.

        Returns the concept dict or None if not found.
        """
        # Try exact match first
        concept = self.bible.lookup_concept(token_text.lower())
        if concept:
            return concept

        # Try lemmatized form
        doc = self.nlp(token_text)
        if doc and doc[0].lemma_ != token_text.lower():
            concept = self.bible.lookup_concept(doc[0].lemma_)
            if concept:
                return concept

        # Fuzzy search fallback
        results = self.bible.search_fuzzy(token_text.lower(), limit=3)
        if not results:
            return None

        # If only one result, use it
        if len(results) == 1:
            return results[0]

        # Disambiguate using sentence embeddings if multiple candidates
        if self.embedder and context:
            import numpy as np
            context_vec = self.embedder.encode(context)
            best_score = -1.0
            best_concept = results[0]
            for candidate in results:
                candidate_text = f"{candidate['surface_text']}: {candidate['definition']}"
                cand_vec = self.embedder.encode(candidate_text)
                score = float(np.dot(context_vec, cand_vec) / (
                    np.linalg.norm(context_vec) * np.linalg.norm(cand_vec) + 1e-8
                ))
                if score > best_score:
                    best_score = score
                    best_concept = candidate
            return best_concept

        return results[0]

    def _make_unknown_eda(self, word: str) -> list:
        """Create an EDA for an unknown concept."""
        anchor = f"unknown_{word.lower().replace(' ', '_')}"
        return [0, 0, 0, 0, anchor, 0, 0, 0.3]

    def _concept_to_eda(self, concept: dict, modifiers: Optional[list[dict]] = None, confidence: float = 0.9) -> list:
        """Convert a Bible concept dict to an EDA array."""
        mod1 = modifiers[0]["anchor_token"] if modifiers and len(modifiers) > 0 else 0
        mod2 = modifiers[1]["anchor_token"] if modifiers and len(modifiers) > 1 else 0
        return [
            concept["domain"],
            concept["category"],
            concept["subcategory"],
            concept["specificity"],
            concept["anchor_token"],
            mod1,
            mod2,
            round(confidence, 2),
        ]

    def _find_relation_type(self, dep_label: str) -> Optional[int]:
        """Map spaCy dependency label to SML relation type ID."""
        # Map common dependency labels to ConceptNet relation types
        dep_to_rel = {
            "nsubj": None,      # Subject — handled by RA subject_ref
            "dobj": None,       # Direct object — handled by RA object_ref
            "prep_in": 6,       # AtLocation
            "prep_on": 6,       # AtLocation
            "prep_at": 6,       # AtLocation
            "prep_to": 6,       # AtLocation
            "amod": 4,          # HasProperty (adjective modifier)
            "attr": 1,          # IsA
            "pobj": 6,          # AtLocation (prep object)
        }
        return dep_to_rel.get(dep_label)

    def encode(self, text: str) -> str:
        """Encode a text string into an SML block.

        Returns the formatted <sml>...</sml> block string.
        """
        doc = self.nlp(text)
        entities = []       # List of EDA arrays
        relations = []      # List of RA arrays
        entity_index = {}   # token index -> entity list index

        # Phase 1: Extract entities from noun chunks
        for chunk in doc.noun_chunks:
            head_text = chunk.root.lemma_.lower()
            concept = self._resolve_concept(head_text, text)

            # Collect modifiers (adjectives attached to the noun)
            modifiers = []
            for token in chunk:
                if token.dep_ in ("amod", "acomp") and token != chunk.root:
                    mod_concept = self._resolve_concept(token.lemma_.lower(), text)
                    if mod_concept:
                        modifiers.append(mod_concept)

            if concept:
                eda = self._concept_to_eda(concept, modifiers)
            else:
                eda = self._make_unknown_eda(head_text)

            idx = len(entities)
            entities.append(eda)
            entity_index[chunk.root.i] = idx

        # Phase 2: Extract relations from dependency tree
        for token in doc:
            # Look for verbs connecting entities
            if token.pos_ == "VERB":
                # Find subject and object
                subj_idx = None
                obj_idx = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and child.i in entity_index:
                        subj_idx = entity_index[child.i]
                    elif child.dep_ in ("dobj", "attr") and child.i in entity_index:
                        obj_idx = entity_index[child.i]

                # Check for prepositional objects
                if obj_idx is None:
                    for child in token.children:
                        if child.dep_ == "prep":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj" and grandchild.i in entity_index:
                                    obj_idx = entity_index[grandchild.i]
                                    break

                if subj_idx is not None and obj_idx is not None:
                    # Determine relation type from verb and prepositions
                    rel_type = 24  # Default: RelatedTo

                    # Check for location prepositions
                    for child in token.children:
                        if child.dep_ == "prep" and child.text.lower() in ("on", "in", "at", "near", "under"):
                            rel_type = 6  # AtLocation
                            break

                    # Check verb-specific mappings
                    verb_rel_map = {
                        "be": 1,        # IsA
                        "have": 3,      # HasA
                        "cause": 7,     # Causes
                        "use": 12,      # UsedFor
                        "make": 13,     # CreatedBy
                        "want": 18,     # Desires
                        "need": 8,      # HasPrerequisite
                        "eat": 12,      # UsedFor (food)
                    }
                    verb_lemma = token.lemma_.lower()
                    if verb_lemma in verb_rel_map:
                        rel_type = verb_rel_map[verb_lemma]

                    # Determine temporal from verb tense
                    temporal = 0  # default: atemporal
                    if token.morph.get("Tense"):
                        tense = token.morph.get("Tense")[0]
                        if tense == "Past":
                            temporal = 1
                        elif tense == "Pres":
                            temporal = 2

                    ra = [rel_type, subj_idx, obj_idx, 0.85, temporal, 0]
                    relations.append(ra)

        # If no entities found, try individual nouns
        if not entities:
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN"):
                    concept = self._resolve_concept(token.lemma_.lower(), text)
                    if concept:
                        eda = self._concept_to_eda(concept)
                    else:
                        eda = self._make_unknown_eda(token.lemma_.lower())
                    entities.append(eda)

        # If still nothing, create an unknown block
        if not entities:
            entities.append(self._make_unknown_eda(text[:20]))

        return format_sml_block(entities, relations)

    def encode_for_training(self, text: str) -> dict:
        """Encode text and return structured data (for training pipeline).

        Returns dict with "sml_block" (formatted string),
        "entities" (raw arrays), "relations" (raw arrays).
        """
        sml_block = self.encode(text)
        # Also parse it back out for validation
        from sml.encoder.formatter import parse_sml_block
        parsed = parse_sml_block(sml_block)
        return {
            "sml_block": sml_block,
            "entities": parsed["entities"],
            "relations": parsed["relations"],
        }
