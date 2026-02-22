#!/usr/bin/env python3
"""Convert existing training data from old format to V3 format.

Old format:
  system: "You are a helpful AI assistant with access to..."
  user: "What color is the sun?"
  assistant: "<sml>E(...)R(...)</sml>\n<think>...</think>\n<response>...</response>"

V3 format:
  system: "You are an AI assistant that uses Structured Markup Language..."
  user: "<sml>E(...)R(...)</sml>\n\nWhat color is the sun?"
  assistant: "<think>...</think>\n\n{answer text}"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sml.config import TRAINING_DATA_PATH, V3_SYSTEM_PROMPT
from sml.training.data_generator_v3 import _validate_v3_example


def _extract_sml_block(text: str) -> str:
    """Extract <sml>...</sml> block from text."""
    match = re.search(r"(<sml>.*?</sml>)", text, re.DOTALL)
    return match.group(1) if match else ""


def _extract_think_block(text: str) -> str:
    """Extract content inside <think>...</think> or <thinking>...</thinking>."""
    # Try <think> first
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try <thinking> variant
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_response_block(text: str) -> str:
    """Extract content inside <response>...</response>."""
    match = re.search(r"<response>(.*?)</response>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: take everything after </think> (or </thinking>)
    for tag in ["</think>", "</thinking>"]:
        if tag in text:
            after = text.split(tag, 1)[1].strip()
            # Strip stray response tags
            after = after.replace("<response>", "").replace("</response>", "").strip()
            if after:
                return after
    return ""


def convert_example(old: dict) -> Optional[dict]:
    """Convert a single training example from old to V3 format.

    Returns None if the example can't be converted.
    """
    msgs = old.get("messages", [])
    if len(msgs) < 3:
        return None

    # Extract question from user message
    question = msgs[1].get("content", "").strip()
    if not question:
        return None

    # Extract assistant content
    assistant_content = msgs[2].get("content", "").strip()
    if not assistant_content:
        return None

    # Parse out SML block from assistant
    sml_block = _extract_sml_block(assistant_content)
    if not sml_block:
        return None

    # Parse out think content
    think_content = _extract_think_block(assistant_content)

    # Parse out response content
    response_content = _extract_response_block(assistant_content)
    if not response_content:
        return None

    # Reassemble in V3 format
    user_content = f"{sml_block}\n\n{question}"

    if think_content:
        assistant_v3 = f"<think>\n{think_content}\n</think>\n\n{response_content}"
    else:
        assistant_v3 = response_content

    return {
        "messages": [
            {"role": "system", "content": V3_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_v3},
        ],
    }


def convert_file(
    input_path: str,
    output_path: str,
    validate: bool = False,
) -> dict:
    """Convert a JSONL file from old format to V3 format.

    Returns stats dict.
    """
    stats = {
        "total": 0,
        "converted": 0,
        "skipped": 0,
        "validation_errors": 0,
    }

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            stats["total"] += 1
            line = line.strip()
            if not line:
                stats["skipped"] += 1
                continue

            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Line {line_num}: JSON parse error, skipping")
                stats["skipped"] += 1
                continue

            converted = convert_example(old)
            if converted is None:
                stats["skipped"] += 1
                continue

            if validate:
                errors = _validate_v3_example(converted)
                if errors:
                    stats["validation_errors"] += 1
                    if stats["validation_errors"] <= 10:
                        print(f"  Line {line_num}: {errors}")
                    continue

            fout.write(json.dumps(converted) + "\n")
            stats["converted"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert existing training data to V3 format"
    )
    parser.add_argument(
        "-i", "--input", type=str, default=str(TRAINING_DATA_PATH),
        help="Input JSONL file (old format)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSONL file (V3 format). Default: input path with _v3_converted suffix",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate each converted example (skip invalid ones)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_v3_converted{p.suffix}")

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("Convert Training Data to V3 Format")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Validate: {args.validate}")
    print()

    stats = convert_file(args.input, args.output, validate=args.validate)

    print(f"\nConversion complete:")
    print(f"  Total examples: {stats['total']}")
    print(f"  Converted: {stats['converted']}")
    print(f"  Skipped: {stats['skipped']}")
    if args.validate:
        print(f"  Validation errors: {stats['validation_errors']}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
