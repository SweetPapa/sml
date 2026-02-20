#!/usr/bin/env python3
"""SML QLoRA Fine-Tuning Script — trains Qwen2.5-3B-Instruct on SML data."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sml.config import DEFAULT_TRAINING_ARGS, TRAINING_DATA_PATH, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-3B on SML training data")
    parser.add_argument("--data", type=str, default=str(TRAINING_DATA_PATH),
                        help="Path to training JSONL file")
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "model_output"),
                        help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_ARGS["num_train_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_ARGS["per_device_train_batch_size"],
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAINING_ARGS["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_TRAINING_ARGS["lora_r"],
                        help="LoRA rank")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_TRAINING_ARGS["max_seq_length"],
                        help="Maximum sequence length")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA weights and save full model after training")
    args = parser.parse_args()

    # Check data exists
    if not Path(args.data).exists():
        print(f"Error: Training data not found at {args.data}")
        print("Run scripts/02_generate_data.py first to generate training data.")
        sys.exit(1)

    print("=" * 60)
    print("SML Fine-Tuning with QLoRA + Unsloth")
    print("=" * 60)
    print(f"Training data: {args.data}")
    print(f"Output dir: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA rank: {args.lora_r}")
    print()

    # Step 1: Load model with Unsloth
    print("Step 1: Loading model with Unsloth...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=DEFAULT_TRAINING_ARGS["model_name"],
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # Step 2: Configure LoRA
    print("Step 2: Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=DEFAULT_TRAINING_ARGS["lora_alpha"],
        target_modules=DEFAULT_TRAINING_ARGS["target_modules"],
        lora_dropout=DEFAULT_TRAINING_ARGS["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Step 3: Load training data
    print("Step 3: Loading training dataset...")
    from sml.training.dataset import load_sml_dataset

    dataset = load_sml_dataset(args.data)

    # Step 4: Setup trainer
    print("Step 4: Setting up SFT Trainer...")
    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=DEFAULT_TRAINING_ARGS["gradient_accumulation_steps"],
        learning_rate=args.lr,
        lr_scheduler_type=DEFAULT_TRAINING_ARGS["lr_scheduler_type"],
        warmup_ratio=DEFAULT_TRAINING_ARGS["warmup_ratio"],
        weight_decay=DEFAULT_TRAINING_ARGS["weight_decay"],
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        report_to="none",
    )

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
    )

    # Step 5: Train
    print("Step 5: Starting training...")
    print()
    trainer.train()

    # Step 6: Save
    print("\nStep 6: Saving model...")
    adapter_path = output_dir / "sml_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved to {adapter_path}")

    # Optionally merge
    if args.merge:
        print("\nStep 7: Merging LoRA weights into base model...")
        merged_path = output_dir / "sml_merged"
        model.save_pretrained_merged(str(merged_path), tokenizer)
        print(f"Merged model saved to {merged_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
