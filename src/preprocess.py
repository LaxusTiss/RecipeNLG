import argparse
import ast
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

RECIPE_TOKEN = "<|recipe|>"


def parse_list_cell(value: Any) -> list[str]:
    """Parse a list-like CSV cell from RecipeNLG into a Python list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(value).strip()]


def format_directions(directions: Iterable[str]) -> str:
    steps = [step.strip() for step in directions if step and step.strip()]
    return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))


def format_recipe(
    row: pd.Series,
    title_column: str = "title",
    ingredients_column: str = "ingredients",
    directions_column: str = "directions",
) -> str:
    title = str(row.get(title_column, "")).strip()
    ingredients = parse_list_cell(row.get(ingredients_column))
    directions = parse_list_cell(row.get(directions_column))

    ingredients_text = ", ".join(ingredients)
    directions_text = format_directions(directions)
    recipe_body = f"{RECIPE_TOKEN} {title} | {ingredients_text}\n{directions_text}".strip()
    return f"{recipe_body}<|endoftext|>"


def load_recipe_dataframe(
    data_path: str | Path,
    num_samples: Optional[int] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    frame = frame.dropna(subset=["title", "ingredients", "directions"]).copy()
    if num_samples is not None and num_samples < len(frame):
        frame = frame.sample(n=num_samples, random_state=random_seed).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    return frame


def build_text_dataset(
    data_path: str | Path,
    num_samples: Optional[int] = None,
    random_seed: int = 42,
) -> Dataset:
    frame = load_recipe_dataframe(data_path, num_samples=num_samples, random_seed=random_seed)
    frame["text"] = frame.apply(format_recipe, axis=1)
    return Dataset.from_pandas(frame[["text"]], preserve_index=False)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess RecipeNLG data for GPT-2 training.")
    parser.add_argument("--data_path", required=True, help="Path to full_dataset.csv")
    parser.add_argument("--output_path", help="Optional path to save formatted data as JSONL")
    parser.add_argument("--tokenizer_name", default="distilgpt2", help="Tokenizer to use for preview tokenization")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of random samples to keep")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length for tokenization preview")
    parser.add_argument("--preview_rows", type=int, default=3, help="How many formatted examples to print")
    args = parser.parse_args()

    dataset = build_text_dataset(args.data_path, num_samples=args.num_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [RECIPE_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loaded {len(dataset)} formatted recipes.")
    for idx in range(min(args.preview_rows, len(dataset))):
        print(f"\n--- Example {idx + 1} ---")
        print(dataset[idx]["text"][:800])

    tokenized = tokenize_dataset(dataset.select(range(min(len(dataset), args.preview_rows))), tokenizer, args.max_len)
    print(f"\nTokenized preview columns: {tokenized.column_names}")
    print(f"First example token count: {sum(1 for token in tokenized[0]['attention_mask'] if token)}")

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in dataset:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved formatted dataset to {output_path}")


if __name__ == "__main__":
    main()
