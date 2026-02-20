"""SML array formatter — converts arrays to compact string format."""


def format_eda(array: list) -> str:
    """Format an Entity Descriptor Array as compact string.

    Input: [domain, category, subcategory, specificity, identity, mod1, mod2, confidence]
    Output: E(1|12|45|0|dog_4527|brown_102|0|0.98)
    """
    if len(array) != 8:
        raise ValueError(f"EDA must have 8 elements, got {len(array)}")
    return f"E({array[0]}|{array[1]}|{array[2]}|{array[3]}|{array[4]}|{array[5]}|{array[6]}|{array[7]})"


def format_ra(array: list) -> str:
    """Format a Relation Array as compact string.

    Input: [rel_type, subject_ref, object_ref, weight, temporal, negation]
    Output: R(6|0|1|0.85|2|0)
    """
    if len(array) != 6:
        raise ValueError(f"RA must have 6 elements, got {len(array)}")
    return f"R({array[0]}|{array[1]}|{array[2]}|{array[3]}|{array[4]}|{array[5]})"


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
            for p in parts:
                try:
                    parsed.append(int(p))
                except ValueError:
                    try:
                        parsed.append(float(p))
                    except ValueError:
                        parsed.append(p)
            relations.append(parsed)

    return {"entities": entities, "relations": relations}
