# Fine-Tuning LLMs with LoRA / QLoRA (RTX 3090 Guide)

This document provides a reference for using Hugging Face `transformers`, `peft`, and `bitsandbytes` to fine-tune large language models locally.

---

## TrainingArguments (Hugging Face Trainer)

| Argument | Description |
|---|---|
| `output_dir` | Directory for checkpoints and logs |
| `overwrite_output_dir` | Overwrite output if it exists |
| `num_train_epochs` | Number of passes over dataset |
| `max_steps` | Fixed step count (overrides epochs if set) |
| `per_device_train_batch_size` | Train batch size per GPU |
| `per_device_eval_batch_size` | Eval batch size per GPU |
| `gradient_accumulation_steps` | Accumulate grads to simulate bigger batch |
| `learning_rate` | Base learning rate (e.g., `2e-4` for LoRA) |
| `weight_decay` | L2 penalty on weights (e.g., `0.01`) |
| `lr_scheduler_type` | LR schedule: `linear`, `cosine`, etc. |
| `warmup_steps` / `warmup_ratio` | Ramp-up phase for LR |
| `max_grad_norm` | Gradient clipping (default 1.0) |
| `fp16` / `bf16` | Mixed precision (bf16 preferred on 3090) |
| `gradient_checkpointing` | Save VRAM by recomputing activations |
| `dataloader_num_workers` | DataLoader workers (2–8) |
| `optim` | Optimizer: `adamw_torch`, `adamw_bnb_8bit` |
| `logging_steps` | Logging frequency |
| `evaluation_strategy` | When to eval: `steps` / `epoch` |
| `eval_steps` | Eval every N steps |
| `save_strategy` | Save frequency: `steps` / `epoch` |
| `save_total_limit` | Keep only recent checkpoints |
| `load_best_model_at_end` | Reload best checkpoint automatically |
| `metric_for_best_model` | Metric to track (e.g., `loss`) |
| `report_to` | Logging backend: `tensorboard`, `wandb` |
| `seed` | Random seed for reproducibility |
| `predict_with_generate` | Use generation during eval |
| `torch_compile` | Enable PyTorch 2.x compile (optional) |
| `deepspeed` | Path to Deepspeed config (if used) |

---

## LoRAConfig (PEFT)

| Argument | Description |
|---|---|
| `r` | Rank of low-rank adapters (capacity) |
| `lora_alpha` | Scaling factor (strength of adapters) |
| `target_modules` | Which layers get LoRA (e.g., `["q_proj","v_proj"]`) |
| `lora_dropout` | Dropout for LoRA path |
| `bias` | Train biases: `"none"`, `"lora_only"`, `"all"` |
| `task_type` | Task hint: `CAUSAL_LM`, `SEQ_2_SEQ_LM` |
| `inference_mode` | If true, adapters only for inference |
| `modules_to_save` | Extra modules to save/train (e.g., `lm_head`) |
| `layers_to_transform` | Specific layer indexes to apply LoRA |
| `init_lora_weights` | Init strategy for adapter weights |
| `use_rslora` | Rank-stabilized LoRA variant |
| `use_dora` | Decomposed LoRA (if supported in PEFT version) |

---

## Minimal Config Example (QLoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
import bitsandbytes as bnb

# Load base model in 4-bit
model_name = "huggyllama/llama-7b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    quantization_config=bnb.nn.Linear4bit.quantization_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Training arguments
args = TrainingArguments(
    output_dir="./lora-llama",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=20,
    save_strategy="steps",
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    report_to="tensorboard"
)

# Example dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1%]")

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()

---

# 📦 Dependencies

To set up everything needed for local LoRA/QLoRA fine-tuning:

```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv add transformers datasets peft bitsandbytes accelerate sentencepiece safetensors tensorboard