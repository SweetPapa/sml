"""SQLite schema creation for the SML Bible."""

import sqlite3
from pathlib import Path


def create_bible_db(db_path: str) -> None:
    """Create and initialize the SML Bible SQLite database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Enable WAL mode and FTS5
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")

    # ── Core tables ──────────────────────────────────────────────────────

    cur.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY,
            uri TEXT UNIQUE NOT NULL,
            surface_text TEXT NOT NULL,
            anchor_token TEXT NOT NULL,
            domain INTEGER NOT NULL DEFAULT 0,
            category INTEGER NOT NULL DEFAULT 0,
            subcategory INTEGER NOT NULL DEFAULT 0,
            specificity INTEGER NOT NULL DEFAULT 0,
            definition TEXT DEFAULT '',
            vector_blob BLOB DEFAULT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relation_type_id INTEGER NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            FOREIGN KEY (source_id) REFERENCES concepts(id),
            FOREIGN KEY (target_id) REFERENCES concepts(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS relation_types (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL,
            inverse_label TEXT DEFAULT ''
        )
    """)

    # ── Indexes ──────────────────────────────────────────────────────────

    cur.execute("CREATE INDEX IF NOT EXISTS idx_concepts_uri ON concepts(uri)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_concepts_surface ON concepts(surface_text)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_concepts_anchor ON concepts(anchor_token)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_relations_source "
        "ON relations(source_id, relation_type_id, target_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_relations_target "
        "ON relations(target_id, relation_type_id, source_id)"
    )

    # ── FTS5 virtual table ───────────────────────────────────────────────

    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
            surface_text, anchor_token, definition,
            content=concepts, content_rowid=id
        )
    """)

    # ── FTS sync triggers ────────────────────────────────────────────────

    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS concepts_ai AFTER INSERT ON concepts BEGIN
            INSERT INTO concepts_fts(rowid, surface_text, anchor_token, definition)
            VALUES (new.id, new.surface_text, new.anchor_token, new.definition);
        END
    """)

    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS concepts_ad AFTER DELETE ON concepts BEGIN
            INSERT INTO concepts_fts(concepts_fts, rowid, surface_text, anchor_token, definition)
            VALUES ('delete', old.id, old.surface_text, old.anchor_token, old.definition);
        END
    """)

    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS concepts_au AFTER UPDATE ON concepts BEGIN
            INSERT INTO concepts_fts(concepts_fts, rowid, surface_text, anchor_token, definition)
            VALUES ('delete', old.id, old.surface_text, old.anchor_token, old.definition);
            INSERT INTO concepts_fts(rowid, surface_text, anchor_token, definition)
            VALUES (new.id, new.surface_text, new.anchor_token, new.definition);
        END
    """)

    # ── Populate relation types (34 ConceptNet relations) ────────────────

    relation_types = [
        (1, "IsA", "HasInstance"),
        (2, "PartOf", "HasPart"),
        (3, "HasA", "PartOf"),
        (4, "HasProperty", "PropertyOf"),
        (5, "CapableOf", "CapabilityOf"),
        (6, "AtLocation", "LocationOf"),
        (7, "Causes", "CausedBy"),
        (8, "HasPrerequisite", "PrerequisiteOf"),
        (9, "HasFirstSubevent", "FirstSubeventOf"),
        (10, "HasLastSubevent", "LastSubeventOf"),
        (11, "MotivatedByGoal", "GoalOf"),
        (12, "UsedFor", "UsedBy"),
        (13, "CreatedBy", "Creates"),
        (14, "DefinedAs", "Defines"),
        (15, "SymbolOf", "SymbolizedBy"),
        (16, "MadeOf", "MaterialOf"),
        (17, "ReceivesAction", "ActionAppliedTo"),
        (18, "Desires", "DesiredBy"),
        (19, "CausesDesire", "DesireCausedBy"),
        (20, "HasContext", "ContextOf"),
        (21, "SimilarTo", "SimilarTo"),
        (22, "Antonym", "Antonym"),
        (23, "DerivedFrom", "DerivesInto"),
        (24, "RelatedTo", "RelatedTo"),
        (25, "FormOf", "HasForm"),
        (26, "EtymologicallyRelatedTo", "EtymologicallyRelatedTo"),
        (27, "Synonym", "Synonym"),
        (28, "MannerOf", "HasManner"),
        (29, "LocatedNear", "LocatedNear"),
        (30, "HasContext", "ContextOf"),
        (31, "dbpedia/genre", ""),
        (32, "dbpedia/occupation", ""),
        (33, "dbpedia/language", ""),
        (34, "dbpedia/capital", ""),
    ]

    cur.executemany(
        "INSERT OR IGNORE INTO relation_types (id, label, inverse_label) VALUES (?, ?, ?)",
        relation_types,
    )

    conn.commit()
    conn.close()
