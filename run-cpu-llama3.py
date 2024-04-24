from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama-3-8b-it")
model = AutoModelForCausalLM.from_pretrained("meta-llama-3-8b-it", torch_dtype=torch.bfloat16)

messages = [
    {"role": "system", "content": "You are a nice tourist friend who always gives tips for whoever asks about vacations"},
    {"role": "user", "content": "List four plans for tourists in California"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
outputs = model.generate(input_ids,
                        max_new_tokens=256,
                        eos_token_id=terminators,
                        do_sample=True,
                        top_p=0.9)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))