from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google-gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained("google-gemma-1.1-2b-it", torch_dtype=torch.bfloat16)

input_text = "List ten plans for tourists in California"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_length=512)
print(tokenizer.decode(outputs[0]))