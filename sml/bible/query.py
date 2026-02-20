"""Bible query interface for the SML Bible."""

import sqlite3
from typing import Optional


class Bible:
    """Read-only query interface to an SML Bible SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def lookup_concept(self, text: str) -> Optional[dict]:
        """Exact match on surface_text (case-insensitive)."""
        row = self.conn.execute(
            "SELECT * FROM concepts WHERE LOWER(surface_text) = LOWER(?)", (text,)
        ).fetchone()
        return dict(row) if row else None

    def lookup_by_anchor(self, anchor: str) -> Optional[dict]:
        """Lookup by anchor_token."""
        row = self.conn.execute(
            "SELECT * FROM concepts WHERE anchor_token = ?", (anchor,)
        ).fetchone()
        return dict(row) if row else None

    def get_relations(self, concept_id: int) -> list[dict]:
        """Get all relations where concept is source or target."""
        rows = self.conn.execute(
            """SELECT r.*, rt.label as relation_label, rt.inverse_label
               FROM relations r
               JOIN relation_types rt ON r.relation_type_id = rt.id
               WHERE r.source_id = ? OR r.target_id = ?""",
            (concept_id, concept_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_outgoing_relations(self, concept_id: int) -> list[dict]:
        """Get relations where concept is the source."""
        rows = self.conn.execute(
            """SELECT r.*, rt.label as relation_label,
                      c.surface_text as target_text, c.anchor_token as target_anchor
               FROM relations r
               JOIN relation_types rt ON r.relation_type_id = rt.id
               JOIN concepts c ON r.target_id = c.id
               WHERE r.source_id = ?""",
            (concept_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_taxonomy(self, concept_id: int) -> dict:
        """Get domain/category/subcategory/specificity for a concept."""
        row = self.conn.execute(
            "SELECT domain, category, subcategory, specificity FROM concepts WHERE id = ?",
            (concept_id,),
        ).fetchone()
        return dict(row) if row else {}

    def search_fuzzy(self, text: str, limit: int = 5) -> list[dict]:
        """FTS5 fuzzy search on surface_text."""
        rows = self.conn.execute(
            """SELECT c.* FROM concepts_fts fts
               JOIN concepts c ON fts.rowid = c.id
               WHERE concepts_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (f"{text}*", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_concept_by_id(self, concept_id: int) -> Optional[dict]:
        """Get a concept by its integer ID."""
        row = self.conn.execute(
            "SELECT * FROM concepts WHERE id = ?", (concept_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_concepts(self) -> list[dict]:
        """Get all concepts (for small/micro Bibles)."""
        rows = self.conn.execute("SELECT * FROM concepts ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def get_relation_type(self, type_id: int) -> Optional[dict]:
        """Get relation type label by ID."""
        row = self.conn.execute(
            "SELECT * FROM relation_types WHERE id = ?", (type_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_concepts(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]

    def count_relations(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
