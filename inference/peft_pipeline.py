import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftConfig, PeftModel

peft_model_id = "../models/lora_20"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
# You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference

batch = tokenizer("<human>: What is B4i?\n<bot>:", return_tensors="pt")

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=128, temperature=0.1)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
