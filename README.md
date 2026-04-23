# 🍳 Recipe Generation with GPT-2 & RecipeNLG

Fine-tune a lightweight GPT-2 model to generate cooking recipes from a dish name and ingredient list — runs fully on **Google Colab free (T4 GPU)**, no paid API required.

---

## 📌 Overview

| | |
|---|---|
| **Task** | Causal Language Modeling (Text Generation) |
| **Model** | `distilgpt2` (82M params) |
| **Dataset** | [RecipeNLG](https://www.kaggle.com/datasets/paultimothymooney/recipenlg) (~2M recipes) |
| **Training subset** | 50,000 recipes |
| **Hardware** | Google Colab T4 (free) |
| **Train time** | ~1.5–2 hours |

**Input → Output example:**

```
Input:  <|recipe|> Garlic Butter Shrimp | shrimp, butter, garlic, lemon, parsley
Output: 1. Melt butter in a skillet over medium heat.
        2. Add garlic and sauté for 1 minute...
        3. Toss in shrimp and cook until pink...
```

---

## 📁 Project Structure

```
recipe-generation/
├── data/
│   └── full_dataset.csv          # Download từ Kaggle
├── notebooks/
│   └── train.ipynb               # Notebook chính để train trên Colab
├── src/
│   ├── preprocess.py             # Format và tokenize dataset
│   ├── train.py                  # Training script
│   └── generate.py               # Inference / sinh công thức
├── outputs/
│   └── recipe-gpt2-final/        # Model sau khi train (lưu vào Drive)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone repo

```bash
git clone https://github.com/your-username/recipe-generation.git
cd recipe-generation
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
torch>=2.0.0
pandas
```

### 3. Tải dataset

Vào [Kaggle RecipeNLG](https://www.kaggle.com/datasets/paultimothymooney/recipenlg), download file `full_dataset.csv` và đặt vào thư mục `data/`.

---

## 🚀 Training

### Chạy trên Google Colab (khuyến nghị)

1. Upload notebook `notebooks/train.ipynb` lên Colab
2. Chọn **Runtime → Change runtime type → T4 GPU**
3. Mount Google Drive để lưu model:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Chạy từng cell theo thứ tự

### Chạy local

```bash
python src/train.py \
  --data_path data/full_dataset.csv \
  --output_dir outputs/recipe-gpt2-final \
  --num_samples 50000 \
  --epochs 3 \
  --batch_size 8 \
  --max_len 512 \
  --fp16
```

---

## 🧠 Data Preprocessing

Mỗi recipe được format thành 1 chuỗi văn bản theo template:

```
<|recipe|> {title} | {ingredient_1}, {ingredient_2}, ...
1. {step_1}
2. {step_2}
...
<|endoftext|>
```

Code xử lý trong `src/preprocess.py`:

```python
def format_recipe(row):
    import ast
    ings = ast.literal_eval(row['ingredients'])
    dirs = ast.literal_eval(row['directions'])
    ings_str = ", ".join(ings)
    dirs_str = " ".join([f"{i+1}. {s}" for i, s in enumerate(dirs)])
    return f"<|recipe|> {row['title']} | {ings_str}\n{dirs_str}<|endoftext|>"
```

---

## 🏋️ Training Config

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=8,   # giảm xuống 4 nếu OOM
    warmup_steps=200,
    weight_decay=0.01,
    fp16=True,                       # bật trên GPU
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

| Hyperparameter | Giá trị |
|---|---|
| Learning rate | 5e-5 (default AdamW) |
| Max sequence length | 512 tokens |
| Batch size | 8 |
| Epochs | 3 |
| Precision | FP16 |

---

## 🍽️ Inference

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

model = GPT2LMHeadModel.from_pretrained("outputs/recipe-gpt2-final")
tokenizer = GPT2Tokenizer.from_pretrained("outputs/recipe-gpt2-final")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

prompt = "<|recipe|> Chocolate Lava Cake | chocolate, butter, eggs, sugar, flour\n"

result = generator(
    prompt,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)

print(result[0]['generated_text'])
```

### Điều chỉnh generation

| Parameter | Thấp | Cao |
|---|---|---|
| `temperature` | Ổn định, lặp lại | Sáng tạo, đa dạng |
| `top_k` | Tập trung | Phong phú hơn |
| `max_new_tokens` | Công thức ngắn | Công thức dài |

---

## 📊 Đánh giá model

Dùng **Perplexity** để đo độ tốt của language model (thấp hơn = tốt hơn):

```python
import torch
import math

def compute_perplexity(model, tokenizer, text, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return math.exp(loss.item())

sample = "<|recipe|> Pancakes | flour, milk, eggs, butter, sugar\n1. Mix dry..."
ppl = compute_perplexity(model, tokenizer, sample)
print(f"Perplexity: {ppl:.2f}")
```

---

## 🖥️ Gradio Demo (tuỳ chọn)

```bash
pip install gradio
```

```python
import gradio as gr

def generate_recipe(title, ingredients):
    prompt = f"<|recipe|> {title} | {ingredients}\n"
    result = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.95)
    return result[0]['generated_text'].replace(prompt, "")

demo = gr.Interface(
    fn=generate_recipe,
    inputs=[
        gr.Textbox(label="Tên món ăn", placeholder="Garlic Butter Shrimp"),
        gr.Textbox(label="Nguyên liệu", placeholder="shrimp, butter, garlic, lemon"),
    ],
    outputs=gr.Textbox(label="Công thức"),
    title="🍳 AI Recipe Generator",
)

demo.launch(share=True)  # share=True để có public URL trên Colab
```

---

## 💡 Cải thiện kết quả

- **Nhiều data hơn:** tăng lên 200k–500k mẫu để model phong phú hơn
- **Model lớn hơn:** dùng `gpt2-medium` (345M) hoặc `gpt2-large` (774M) nếu có GPU mạnh hơn
- **LoRA / PEFT:** fine-tune hiệu quả hơn với VRAM thấp

```bash
pip install peft
```

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 294,912 / 81,912,576 — chỉ train ~0.36%!
```

- **Domain cụ thể:** lọc dataset theo cuisine (Italian, Vietnamese, ...) để model chuyên biệt hơn

---

## 📜 License

MIT License. Dataset RecipeNLG thuộc license riêng — xem [tại đây](https://recipenlg.cs.put.poznan.pl/).

---

## 🙏 References

- [RecipeNLG Paper](https://aclanthology.org/2020.inlg-1.4/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [distilgpt2 Model Card](https://huggingface.co/distilgpt2)