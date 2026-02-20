"""HuggingFace Dataset loader for SML training data."""
import json
from typing import Optional

from sml.config import TRAINING_DATA_PATH


def load_sml_dataset(
    data_path: Optional[str] = None,
    test_size: float = 0.1,
    seed: int = 42,
):
    """Load validated JSONL training data into a HuggingFace Dataset.

    Args:
        data_path: Path to JSONL file with ChatML training examples.
        test_size: Fraction of data to hold out for validation.
        seed: Random seed for splitting.

    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    from datasets import Dataset, DatasetDict

    data_path = data_path or str(TRAINING_DATA_PATH)

    # Load JSONL
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if not examples:
        raise ValueError(f"No examples found in {data_path}")

    # Extract the text field for training: we store the full messages list
    # The trainer (SFTTrainer) expects either a "text" field or "messages" field
    records = []
    for ex in examples:
        records.append({"messages": ex["messages"]})

    dataset = Dataset.from_list(records)

    # Split into train/test
    split = dataset.train_test_split(test_size=test_size, seed=seed)

    print(f"Loaded {len(examples)} examples from {data_path}")
    print(f"  Train: {len(split['train'])}, Test: {len(split['test'])}")

    return split


def format_for_sft(example: dict, tokenizer) -> str:
    """Format a single example for SFT training using the tokenizer's chat template.

    This applies the ChatML template to produce the full training text.
    """
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
