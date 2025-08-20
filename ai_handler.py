from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def generate():
    base_model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",  # or "cuda" if using GPU
    )

    adapter_path = "./adapter_model.safetensors"  # your LoRA file
    model = PeftModel.from_pretrained(model, adapter_path)

    prompt = "Explain quantum computing in simple terms."

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)