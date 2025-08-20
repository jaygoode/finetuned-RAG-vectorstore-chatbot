from transformers import AutoModelForCasualLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

#load basemodel
model_name = "mistralai/Mistral-78-v0.1"
model = AutoModelForCasualLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

#define LoRA config

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSUAL_LM"
)

#wrap base model with LoRA
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files={"train": "train.json", "test": "test.json"})

training_args = TrainingArguments(
    output_dir="./lora_out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    save_steps="500",
    save_total_limit=2,
    num_train_epochs=3
)

#trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)
trainer.train()

model.save_pretrained("./lora-adapter")

