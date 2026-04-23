import argparse

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from generate import build_prompt


class Translator:
    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device

    def translate(self, text: str, max_new_tokens: int = 256) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""

        inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def load_generator(model_dir: str, force_cpu: bool = False):
    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    vi_to_en = Translator("Helsinki-NLP/opus-mt-vi-en", device)
    en_to_vi = Translator("Helsinki-NLP/opus-mt-en-vi", device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_recipe(
        title: str,
        ingredients: str,
        input_language: str,
        output_language: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, str]:
        normalized_title = title.strip()
        normalized_ingredients = ingredients.strip()

        if input_language == "Tiếng Việt":
            english_title = vi_to_en.translate(normalized_title)
            english_ingredients = vi_to_en.translate(normalized_ingredients)
        else:
            english_title = normalized_title
            english_ingredients = normalized_ingredients

        prompt = build_prompt(english_title, english_ingredients)
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
        english_recipe = generated.replace(prompt, "", 1).replace("<|endoftext|>", "").strip()

        if output_language == "Tiếng Việt":
            displayed_recipe = en_to_vi.translate(english_recipe, max_new_tokens=max_new_tokens * 2)
        else:
            displayed_recipe = english_recipe

        translated_prompt = (
            f"Title (EN): {english_title}\n"
            f"Ingredients (EN): {english_ingredients}"
        )
        return displayed_recipe, translated_prompt

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
            gr.Textbox(label="Tên món ăn", placeholder="Tôm sốt bơ tỏi"),
            gr.Textbox(label="Nguyên liệu", placeholder="tôm, bơ, tỏi, chanh, rau mùi"),
            gr.Radio(
                choices=["Tiếng Việt", "English"],
                value="Tiếng Việt",
                label="Ngôn ngữ đầu vào",
            ),
            gr.Radio(
                choices=["Tiếng Việt", "English"],
                value="Tiếng Việt",
                label="Ngôn ngữ đầu ra",
            ),
            gr.Slider(label="Độ dài công thức", minimum=64, maximum=512, step=16, value=200),
            gr.Slider(label="Temperature", minimum=0.2, maximum=1.5, step=0.05, value=0.8),
            gr.Slider(label="Top-p", minimum=0.5, maximum=1.0, step=0.01, value=0.95),
        ],
        outputs=[
            gr.Textbox(label="Công thức được tạo"),
            gr.Textbox(label="Prompt tiếng Anh nội bộ"),
        ],
        title="AI Recipe Generator",
        description="Nhập tiếng Việt hoặc tiếng Anh. Ứng dụng sẽ tự dịch sang tiếng Anh cho model và có thể dịch ngược kết quả về tiếng Việt.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
