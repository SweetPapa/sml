"""Training data validator — ensures SML blocks match the Bible."""
import json
from pathlib import Path
from typing import Optional

from sml.bible.query import Bible
from sml.encoder.formatter import parse_sml_block
from sml.config import EDA_WIDTH, RA_WIDTH


def validate_training_data(
    data_path: str,
    bible_path: str,
    strict: bool = False,
) -> dict:
    """Validate a JSONL training data file against the Bible.

    Checks:
    - All concept anchor tokens exist in Bible
    - EDA arrays have correct width (8)
    - RA arrays have correct width (6)
    - Relation subject/object refs are valid entity indices
    - Confidence values are in [0, 1] range

    Returns dict with validation stats.
    """
    bible = Bible(bible_path)

    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "warnings": [],
    }

    with open(data_path) as f:
        for line_num, line in enumerate(f, 1):
            stats["total"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                stats["invalid"] += 1
                stats["errors"].append(f"Line {line_num}: Invalid JSON: {e}")
                continue

            errors = _validate_example(example, bible, line_num)
            # Separate warnings from actual errors
            actual_errors = [e for e in errors if "WARNING:" not in e]
            example_warnings = [e for e in errors if "WARNING:" in e]
            stats["warnings"].extend(example_warnings)
            if actual_errors:
                stats["invalid"] += 1
                if strict:
                    stats["errors"].extend(actual_errors)
                else:
                    stats["errors"].append(actual_errors[0])
            else:
                stats["valid"] += 1

    bible.close()

    # Summary
    stats["valid_pct"] = round(100 * stats["valid"] / max(stats["total"], 1), 1)
    print(f"Validation: {stats['valid']}/{stats['total']} valid ({stats['valid_pct']}%)")
    if stats["errors"]:
        print(f"First errors: {stats['errors'][:5]}")
    if stats["warnings"]:
        print(f"Thinking block warnings: {len(stats['warnings'])}")
        for w in stats["warnings"][:3]:
            print(f"  {w}")

    return stats


def _validate_example(example: dict, bible: Bible, line_num: int) -> list[str]:
    """Validate a single training example. Returns list of error strings."""
    errors = []

    # Check structure
    if "messages" not in example:
        return [f"Line {line_num}: Missing 'messages' key"]

    messages = example["messages"]
    if len(messages) < 3:
        return [f"Line {line_num}: Need at least 3 messages (system, user, assistant)"]

    assistant_msg = messages[-1]
    if assistant_msg.get("role") != "assistant":
        return [f"Line {line_num}: Last message must be assistant role"]

    content = assistant_msg.get("content", "")

    # Check for SML block
    if "<sml>" not in content or "</sml>" not in content:
        return [f"Line {line_num}: Missing <sml> block in assistant response"]

    # Check for thinking and response blocks
    if "<thinking>" not in content or "</thinking>" not in content:
        errors.append(f"Line {line_num}: Missing <thinking> block")
    if "<response>" not in content or "</response>" not in content:
        errors.append(f"Line {line_num}: Missing <response> block")

    # Check thinking block quality (warnings, not errors)
    warnings = []
    if "<thinking>" in content and "</thinking>" in content:
        think_start = content.index("<thinking>") + len("<thinking>")
        think_end = content.index("</thinking>")
        thinking_text = content[think_start:think_end].strip()

        # Check minimum length (>= 20 tokens)
        thinking_tokens = thinking_text.split()
        if len(thinking_tokens) < 20:
            warnings.append(
                f"Line {line_num}: WARNING: <thinking> block too short "
                f"({len(thinking_tokens)} tokens, need >=20)"
            )

        # Check that thinking references at least one SML anchor token
        # Extract anchor tokens from SML block
        if "<sml>" in content and "</sml>" in content:
            sml_start_tmp = content.index("<sml>")
            sml_end_tmp = content.index("</sml>") + len("</sml>")
            sml_text_tmp = content[sml_start_tmp:sml_end_tmp]
            # Find all anchor tokens (strings containing _ followed by digits)
            import re as _re
            anchors = _re.findall(r'[a-z]+_\d+', sml_text_tmp)
            if anchors:
                has_anchor_ref = any(a in thinking_text for a in anchors)
                if not has_anchor_ref:
                    warnings.append(
                        f"Line {line_num}: WARNING: <thinking> block does not reference "
                        f"any SML anchor tokens"
                    )

    # Extract and validate SML
    sml_start = content.index("<sml>")
    sml_end = content.index("</sml>") + len("</sml>")
    sml_text = content[sml_start:sml_end]

    parsed = parse_sml_block(sml_text)

    # Validate entities
    for i, eda in enumerate(parsed["entities"]):
        if len(eda) != EDA_WIDTH:
            errors.append(f"Line {line_num}: Entity {i} has width {len(eda)}, expected {EDA_WIDTH}")
            continue

        # Check confidence range
        conf = eda[7]
        if isinstance(conf, (int, float)) and not (0 <= conf <= 1):
            errors.append(f"Line {line_num}: Entity {i} confidence {conf} out of [0,1] range")

        # Check anchor token exists in Bible (skip unknowns)
        anchor = eda[4]
        if isinstance(anchor, str) and not anchor.startswith("unknown_"):
            concept = bible.lookup_by_anchor(anchor)
            if concept is None:
                errors.append(f"Line {line_num}: Entity {i} anchor '{anchor}' not in Bible")

    # Validate relations
    num_entities = len(parsed["entities"])
    for i, ra in enumerate(parsed["relations"]):
        if len(ra) != RA_WIDTH:
            errors.append(f"Line {line_num}: Relation {i} has width {len(ra)}, expected {RA_WIDTH}")
            continue

        # Check subject/object refs are valid entity indices
        subj_ref = ra[1]
        obj_ref = ra[2]
        if isinstance(subj_ref, int) and (subj_ref < 0 or subj_ref >= num_entities):
            errors.append(f"Line {line_num}: Relation {i} subject_ref {subj_ref} out of range [0, {num_entities})")
        if isinstance(obj_ref, int) and (obj_ref < 0 or obj_ref >= num_entities):
            errors.append(f"Line {line_num}: Relation {i} object_ref {obj_ref} out of range [0, {num_entities})")

    return errors


def filter_valid(data_path: str, bible_path: str, output_path: Optional[str] = None) -> str:
    """Filter training data, keeping only valid examples.

    Returns path to filtered output file.
    """
    if output_path is None:
        p = Path(data_path)
        output_path = str(p.parent / f"{p.stem}_validated{p.suffix}")

    bible = Bible(bible_path)
    valid_count = 0
    total_count = 0

    with open(data_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            total_count += 1
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue

            errors = _validate_example(example, bible, line_num)
            if not errors:
                fout.write(json.dumps(example) + "\n")
                valid_count += 1

    bible.close()
    print(f"Filtered: {valid_count}/{total_count} valid examples written to {output_path}")
    return output_path
