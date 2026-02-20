"""SML array formatter — converts arrays to compact string format."""

from sml.config import RELATION_TYPES, RELATION_TYPES_INV


def format_eda(array: list) -> str:
    """Format an Entity Descriptor Array as compact string.

    Input: [domain, category, subcategory, specificity, identity, mod1, mod2, confidence]
    Output: E(1|12|45|0|dog_4527|brown_102|0|0.98)
    """
    if len(array) != 8:
        raise ValueError(f"EDA must have 8 elements, got {len(array)}")
    return f"E({array[0]}|{array[1]}|{array[2]}|{array[3]}|{array[4]}|{array[5]}|{array[6]}|{array[7]})"


def format_ra(array: list) -> str:
    """Format a Relation Array as compact string with string labels.

    Input: [rel_type, subject_ref, object_ref, weight, temporal, negation]
    - rel_type can be int (looked up in RELATION_TYPES) or string
    - negation flag: if 1/True, prepend NOT_ to the relation label

    Output: R(CapableOf|0|1|0.85|2|0) or R(NOT_CapableOf|0|1|0.85|2|0)
    """
    if len(array) != 6:
        raise ValueError(f"RA must have 6 elements, got {len(array)}")

    rel_type = array[0]
    negation = array[5]

    # Resolve numeric rel_type to string label
    if isinstance(rel_type, int):
        label = RELATION_TYPES.get(rel_type, str(rel_type))
    else:
        label = str(rel_type)

    # Prepend NOT_ if negation flag is set
    if negation and str(negation) not in ("0", "0.0", "False"):
        label = f"NOT_{label}"

    return f"R({label}|{array[1]}|{array[2]}|{array[3]}|{array[4]}|0)"


def format_sml_block(entities: list[list], relations: list[list]) -> str:
    """Format complete SML block with entities and relations.

    Returns a string like:
    <sml>
    E(1|12|45|0|dog_4527|brown_102|0|0.98)
    E(1|2|0|0|mat_2002|0|0|0.90)
    R(6|0|1|0.85|2|0)
    </sml>
    """
    lines = []
    for eda in entities:
        lines.append(format_eda(eda))
    for ra in relations:
        lines.append(format_ra(ra))
    inner = "\n".join(lines)
    return f"<sml>\n{inner}\n</sml>"


def parse_sml_block(sml_text: str) -> dict:
    """Parse an SML block back into entities and relations lists.

    Returns {"entities": [...], "relations": [...]}
    """
    entities = []
    relations = []

    # Strip <sml> tags
    content = sml_text.strip()
    if content.startswith("<sml>"):
        content = content[5:]
    if content.endswith("</sml>"):
        content = content[:-6]

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("E(") and line.endswith(")"):
            inner = line[2:-1]
            parts = inner.split("|")
            # Parse each part: try int, then float, then keep as string
            parsed = []
            for p in parts:
                try:
                    parsed.append(int(p))
                except ValueError:
                    try:
                        parsed.append(float(p))
                    except ValueError:
                        parsed.append(p)
            entities.append(parsed)
        elif line.startswith("R(") and line.endswith(")"):
            inner = line[2:-1]
            parts = inner.split("|")
            parsed = []
            negation = 0
            for idx, p in enumerate(parts):
                if idx == 0:
                    # First element is relation type — detect NOT_ prefix
                    rel_label = p
                    if p.startswith("NOT_"):
                        negation = 1
                        rel_label = p[4:]
                    # Reverse-lookup label to ID
                    if rel_label in RELATION_TYPES_INV:
                        parsed.append(RELATION_TYPES_INV[rel_label])
                    else:
                        try:
                            parsed.append(int(rel_label))
                        except ValueError:
                            parsed.append(rel_label)
                else:
                    try:
                        parsed.append(int(p))
                    except ValueError:
                        try:
                            parsed.append(float(p))
                        except ValueError:
                            parsed.append(p)
            # Ensure negation flag is correctly set (position 5)
            if len(parsed) == 6:
                parsed[5] = negation
            relations.append(parsed)

    return {"entities": entities, "relations": relations}
