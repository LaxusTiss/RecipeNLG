import argparse

import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from generate import build_prompt


def load_generator(model_dir: str, force_cpu: bool = False):
    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_recipe(
        title: str,
        ingredients: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        prompt = build_prompt(title, ingredients)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        return generated.replace(prompt, "", 1).replace("<|endoftext|>", "").strip()

    return generate_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for recipe generation.")
    parser.add_argument("--model_dir", required=True, help="Path to the fine-tuned model")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio URL")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    generator = load_generator(args.model_dir, force_cpu=args.no_cuda)

    demo = gr.Interface(
        fn=generator,
        inputs=[
            gr.Textbox(label="Ten mon an", placeholder="Garlic Butter Shrimp"),
            gr.Textbox(label="Nguyen lieu", placeholder="shrimp, butter, garlic, lemon"),
            gr.Slider(label="Max new tokens", minimum=64, maximum=512, step=16, value=200),
            gr.Slider(label="Temperature", minimum=0.2, maximum=1.5, step=0.05, value=0.8),
            gr.Slider(label="Top-p", minimum=0.5, maximum=1.0, step=0.01, value=0.95),
        ],
        outputs=gr.Textbox(label="Cong thuc"),
        title="AI Recipe Generator",
        description="Sinh cong thuc nau an tu ten mon va danh sach nguyen lieu.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
