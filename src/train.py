import argparse
import math
import os
from pathlib import Path

import torch
from datasets import DatasetDict

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

from preprocess import RECIPE_TOKEN, build_text_dataset, tokenize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune distilgpt2 for recipe generation.")
    parser.add_argument("--data_path", required=True, help="Path to RecipeNLG CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints and final model")
    parser.add_argument("--model_name", default="distilgpt2", help="Base model name")
    parser.add_argument(
        "--cache_dir",
        default=".cache/huggingface",
        help="Directory for Hugging Face model/tokenizer cache",
    )
    parser.add_argument("--num_samples", type=int, default=50000, help="How many samples to train on")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--eval_split", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 training on supported GPUs")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum checkpoints to keep")
    return parser.parse_args()


def compute_perplexity(eval_loss: float) -> float:
    try:
        return math.exp(eval_loss)
    except OverflowError:
        return float("inf")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir.resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir.resolve())
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.resolve())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=str(cache_dir))
    tokenizer.add_special_tokens({"additional_special_tokens": [RECIPE_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.model_name, cache_dir=str(cache_dir))
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    raw_dataset = build_text_dataset(args.data_path, num_samples=args.num_samples, random_seed=args.seed)
    split = raw_dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
    dataset = DatasetDict(
        train=tokenize_dataset(split["train"], tokenizer, args.max_len),
        validation=tokenize_dataset(split["test"], tokenizer, args.max_len),
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    use_fp16 = args.fp16 and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        fp16=use_fp16,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        metrics["perplexity"] = compute_perplexity(metrics["eval_loss"])

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    print(f"Training finished. Model saved to: {output_dir.resolve()}")
    if "perplexity" in metrics:
        print(f"Validation perplexity: {metrics['perplexity']:.2f}")


if __name__ == "__main__":
    main()
