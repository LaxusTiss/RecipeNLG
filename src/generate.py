import argparse
import os
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from preprocess import RECIPE_TOKEN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate recipes from a fine-tuned GPT-2 model.")
    parser.add_argument("--model_dir", required=True, help="Path to the trained model directory")
    parser.add_argument(
        "--cache_dir",
        default=".cache/huggingface",
        help="Directory for Hugging Face cache when loading remote models",
    )
    parser.add_argument("--title", required=True, help="Dish title")
    parser.add_argument("--ingredients", required=True, help="Comma-separated ingredient list")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Penalty for repeated text")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU inference")
    return parser.parse_args()


def build_prompt(title: str, ingredients: str) -> str:
    return f"{RECIPE_TOKEN} {title.strip()} | {ingredients.strip()}\n"


def main() -> None:
    args = parse_args()
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir.resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir.resolve())
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.resolve())

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir, cache_dir=str(cache_dir))
    model = GPT2LMHeadModel.from_pretrained(args.model_dir, cache_dir=str(cache_dir)).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = build_prompt(args.title, args.ingredients)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    recipe_text = generated_text.replace(prompt, "", 1).replace("<|endoftext|>", "").strip()

    print("Prompt:")
    print(prompt)
    print("Generated recipe:")
    print(recipe_text)


if __name__ == "__main__":
    main()
