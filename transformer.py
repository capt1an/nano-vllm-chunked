from transformers import AutoModel

model_name = "Qwen/Qwen3-0.6B"
model = AutoModel.from_pretrained(model_name)

print(model)