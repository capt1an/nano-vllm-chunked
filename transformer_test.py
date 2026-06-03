from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model_name = "Qwen/Qwen3-0.6B"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model_name = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map="auto"
)

print(model)

# prompt = "Introduce yourself."

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.6)

# print(tokenizer.decode(outputs[0]))
