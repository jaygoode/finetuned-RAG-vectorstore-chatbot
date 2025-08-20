from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv("ACCESS_TOKEN")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True 
)

#load basemodel
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=access_token
)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
tokenizer.pad_token = tokenizer.eos_token

#define LoRA config

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

#wrap base model with LoRA
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files={"train": "train.json", "test": "test.json"})

def tokenize_fn(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    # For causal LM, labels = input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Make sure to remove original columns if needed and keep 'labels'
tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print(tokenized_dataset["train"].column_names)
breakpoint()

training_args = TrainingArguments(
    output_dir="./lora_out",
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=3
)

#trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)
trainer.train()

model.save_pretrained("./lora-adapter")

