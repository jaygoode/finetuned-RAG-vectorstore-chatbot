# Fine-Tuning LLMs with LoRA / QLoRA (RTX 3090 Guide)

This document provides a reference for using Hugging Face `transformers`, `peft`, and `bitsandbytes` to fine-tune large language models locally.

---

## TrainingArguments (Hugging Face Trainer)

| Argument                        | Description                                | Beginner-Friendly Explanation                                      |
| ------------------------------- | ------------------------------------------ | ------------------------------------------------------------------ |
| `output_dir`                    | Directory for checkpoints and logs         | Where all saved models and logs will go                            |
| `overwrite_output_dir`          | Overwrite output if it exists              | Replace previous saved models if folder already exists             |
| `num_train_epochs`              | Number of passes over dataset              | How many times the model will see your dataset                     |
| `max_steps`                     | Fixed step count (overrides epochs if set) | Stop training after this many steps instead of full epochs         |
| `per_device_train_batch_size`   | Train batch size per GPU                   | Number of samples processed at once per GPU                        |
| `per_device_eval_batch_size`    | Eval batch size per GPU                    | Number of samples used during evaluation at once                   |
| `gradient_accumulation_steps`   | Accumulate grads to simulate bigger batch  | Pretend batch is bigger by adding gradients over multiple steps    |
| `learning_rate`                 | Base learning rate (e.g., `2e-4` for LoRA) | How fast the model updates weights                                 |
| `weight_decay`                  | L2 penalty on weights (e.g., `0.01`)       | Helps prevent overfitting by shrinking weights slightly            |
| `lr_scheduler_type`             | LR schedule: `linear`, `cosine`, etc.      | Controls how learning rate changes over time                       |
| `warmup_steps` / `warmup_ratio` | Ramp-up phase for LR                       | Start training slowly to avoid sudden jumps in learning            |
| `max_grad_norm`                 | Gradient clipping (default 1.0)            | Stops gradients from getting too big, prevents exploding gradients |
| `fp16` / `bf16`                 | Mixed precision (bf16 preferred on 3090)   | Use half precision to save GPU memory and speed up training        |
| `gradient_checkpointing`        | Save VRAM by recomputing activations       | Saves memory by not storing all intermediate results               |
| `dataloader_num_workers`        | DataLoader workers (2–8)                   | How many CPU threads load data at once                             |
| `optim`                         | Optimizer: `adamw_torch`, `adamw_bnb_8bit` | Method for updating model weights                                  |
| `logging_steps`                 | Logging frequency                          | How often to print loss or metrics                                 |
| `evaluation_strategy`           | When to eval: `steps` / `epoch`            | When to check model performance during training                    |
| `eval_steps`                    | Eval every N steps                         | How often to run evaluation                                        |
| `save_strategy`                 | Save frequency: `steps` / `epoch`          | When to save a checkpoint                                          |
| `save_total_limit`              | Keep only recent checkpoints               | Remove older checkpoints to save space                             |
| `load_best_model_at_end`        | Reload best checkpoint automatically       | After training, load the checkpoint with best performance          |
| `metric_for_best_model`         | Metric to track (e.g., `loss`)             | Decide which checkpoint is “best” using this metric                |
| `report_to`                     | Logging backend: `tensorboard`, `wandb`    | Where to send logs (visualization tools)                           |
| `seed`                          | Random seed for reproducibility            | Ensures same results every time you run                            |
| `predict_with_generate`         | Use generation during eval                 | Evaluate by generating text, not just loss                         |
| `torch_compile`                 | Enable PyTorch 2.x compile (optional)      | Speeds up training using PyTorch’s new compiler                    |
| `deepspeed`                     | Path to Deepspeed config (if used)         | Advanced multi-GPU optimization (optional)                         |

---

## LoRAConfig (PEFT)

| Argument              | Description                                         | Beginner-Friendly Explanation                              |
| --------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| `r`                   | Rank of low-rank adapters (capacity)                | How big the small “adapter” layers are                     |
| `lora_alpha`          | Scaling factor (strength of adapters)               | Multiplies adapter impact, like a volume knob              |
| `target_modules`      | Which layers get LoRA (e.g., `["q_proj","v_proj"]`) | Pick which parts of the model to fine-tune                 |
| `lora_dropout`        | Dropout for LoRA path                               | Randomly ignores some adapter weights to avoid overfitting |
| `bias`                | Train biases: `"none"`, `"lora_only"`, `"all"`      | Decide which bias terms to update                          |
| `task_type`           | Task hint: `CAUSAL_LM`, `SEQ_2_SEQ_LM`              | Tells LoRA what type of language task it’s handling        |
| `inference_mode`      | If true, adapters only for inference                | Only use LoRA for generating text, no training             |
| `modules_to_save`     | Extra modules to save/train (e.g., `lm_head`)       | Save certain layers that are normally frozen               |
| `layers_to_transform` | Specific layer indexes to apply LoRA                | Only apply LoRA to some layers, not all                    |
| `init_lora_weights`   | Init strategy for adapter weights                   | How the adapter weights start (random, etc.)               |
| `use_rslora`          | Rank-stabilized LoRA variant                        | More stable training variant of LoRA                       |
| `use_dora`            | Decomposed LoRA (if supported in PEFT version)      | Experimental LoRA variant to reduce memory                 |

---

## Minimal Config Example (QLoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
import bitsandbytes as bnb

# Load base model in 4-bit precision (saves GPU memory)
model_name = "huggyllama/llama-7b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,              # Enable 4-bit quantization
    device_map="auto",             # Automatically map model to GPU
    quantization_config=bnb.nn.Linear4bit.quantization_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",      # Type of 4-bit quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Computation in bf16
        bnb_4bit_use_double_quant=True
    )
)

# Load tokenizer and pad token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
peft_config = LoraConfig(
    r=16,                           # Adapter rank (size)
    lora_alpha=32,                  # Strength of adapters
    target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA
    lora_dropout=0.05,              # Dropout to prevent overfitting
    task_type="CAUSAL_LM"         # Type of language model task
)
model = get_peft_model(model, peft_config)  # Apply LoRA to model

# Training arguments
args = TrainingArguments(
    output_dir="./lora-llama",       # Where to save checkpoints
    per_device_train_batch_size=2,     # Batch size per GPU
    gradient_accumulation_steps=8,     # Simulate bigger batch size
    num_train_epochs=3,                # Number of passes over dataset
    learning_rate=2e-4,                # Learning rate
    bf16=True,                          # Use mixed precision bf16
    logging_steps=20,                  # How often to log
    save_strategy="steps",            # Save checkpoints every N steps
    save_steps=200,
    evaluation_strategy="steps",      # Evaluate every N steps
    eval_steps=200,
    gradient_checkpointing=True,        # Save memory
    optim="adamw_bnb_8bit",           # Optimizer
    report_to="tensorboard"           # Logging backend
)

# Example small dataset for testing
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1%]")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```

---

## 📦 Dependencies

Install everything needed for LoRA/QLoRA fine-tuning locally:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft bitsandbytes accelerate sentencepiece safetensors tensorboard
```
