from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import torch
import transformers
from accelerate import init_empty_weights, infer_auto_device_map
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, AutoConfig, \
    BitsAndBytesConfig


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="imdb", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=1024, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )


# parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
#
# model_args, data_args, training_args = parser.parse_args_into_dataclasses()
config = AutoConfig.from_pretrained("togethercomputer/Pythia-Chat-Base-7B")

with init_empty_weights():
    dummy_model = AutoModelForCausalLM.from_config(config)

device_map = {'gpt_neox.embed_in': 0, 'gpt_neox.layers.0': 0, 'gpt_neox.layers.1': 0, 'gpt_neox.layers.2': 0, 'gpt_neox.layers.3': 0, 'gpt_neox.layers.4': 'disk', 'gpt_neox.layers.5': 'disk', 'gpt_neox.layers.6': 'disk', 'gpt_neox.layers.7': 'cpu', 'gpt_neox.layers.8': 'cpu', 'gpt_neox.layers.9': 'cpu', 'gpt_neox.layers.10': 'cpu', 'gpt_neox.layers.11': 'cpu', 'gpt_neox.layers.12': 'cpu', 'gpt_neox.layers.13': 'cpu', 'gpt_neox.layers.14': 'cpu', 'gpt_neox.layers.15': 'cpu', 'gpt_neox.layers.16': 'cpu', 'gpt_neox.layers.17': 'cpu', 'gpt_neox.layers.18': 'cpu', 'gpt_neox.layers.19': 'cpu', 'gpt_neox.layers.20': 'cpu', 'gpt_neox.layers.21': 'disk', 'gpt_neox.layers.22': 'disk', 'gpt_neox.layers.23': 'disk', 'gpt_neox.layers.24': 'disk', 'gpt_neox.layers.25': 'disk', 'gpt_neox.layers.26': 'disk', 'gpt_neox.layers.27': 'disk', 'gpt_neox.layers.28': 'disk', 'gpt_neox.layers.29': 'disk', 'gpt_neox.layers.30': 'disk', 'gpt_neox.layers.31': 'disk', 'gpt_neox.final_layer_norm': 'disk', 'embed_out': 'disk'}
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

# device_map = {'model.embed_tokens': 'disk', 'model.layers.0': 'disk', 'model.layers.1': 'disk', 'model.layers.2': 'disk', 'model.layers.3': 'disk', 'model.layers.4': 'disk', 'model.layers.5': 'disk', 'model.layers.6': 'disk', 'model.layers.7': 'cpu', 'model.layers.8': 'cpu', 'model.layers.9': 'cpu', 'model.layers.10': 'cpu', 'model.layers.11': 'cpu', 'model.layers.12': 'cpu', 'model.layers.13': 'cpu', 'model.layers.14': 'cpu', 'model.layers.15': 'cpu', 'model.layers.16': 'cpu', 'model.layers.17': 'cpu', 'model.layers.18': 'cpu', 'model.layers.19': 'disk', 'model.layers.20': 'disk', 'model.layers.21': 'disk', 'model.layers.22': 'disk', 'model.layers.23': 'disk', 'model.layers.24': 'disk', 'model.layers.25': 'disk', 'model.layers.26': 'disk', 'model.layers.27': 'disk', 'model.layers.28': 'disk', 'model.layers.29': 'disk', 'model.layers.30': 'disk', 'model.layers.31': 'disk', 'model.norm': 'disk', 'lm_head': 'disk'}

print(f'Device Map: {device_map}')

model = AutoModelForCausalLM.from_pretrained(
    "togethercomputer/Pythia-Chat-Base-7B",
    load_in_8bit=True,
    # device_map=device_map,
    # quantization_config=quantization_config,
    # offload_folder="./offload_tmp",
    # offload_state_dict=True
    device_map="auto",
    torch_dtype=torch.int8
)
print("Footprint:")
print(model.get_memory_footprint())

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Pythia-Chat-Base-7B")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
# ### Prepare model for training
#
# Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_int8_training` that will:
# - Cast the layer norm in `float32` for stability purposes
# - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
# - Enable gradient checkpointing for more memory-efficient training
# - Cast the output logits in `float32` for smoother sampling during the sampling procedure


# if "gpt-neox" in model_args.model_name_or_path:
model = prepare_model_for_int8_training(
    model, output_embedding_layer_name="embed_out", layer_norm_names=["layer_norm", "layernorm"]
)
# else:
#     model = prepare_model_for_int8_training(model)


# ### Apply LoRA
#
# Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# target_modules = None
# if "gpt-neox" in model_args.model_name_or_path:
target_modules = ["query_key_value", "xxx"]  # workaround to use 8bit training on this model
config = LoraConfig(
    r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

block_size = 1024


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# ### Training
raw_datasets = load_dataset("json", data_dir="../datasets")
data = raw_datasets.remove_columns("metadata")
columns = data["train"].features
data = data.map(lambda samples: tokenizer(samples["text"]), batched=True, remove_columns=columns)
data = data.map(group_texts, batched=True)

model.gradient_checkpointing_enable()
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
            output_dir=f"../models/BocconiGPT_LoRa/",
            num_train_epochs=1,
            push_to_hub=False,
        ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.save_model()

# ## Share adapters on the 🤗 Hub
# model.push_to_hub(training_args.output_dir, use_auth_token=True)
#
# # Load adapters from the Hub and generate some output texts:
#
# peft_model_id = training_args.output_dir
# config = PeftConfig.from_pretrained(peft_model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#
# # Load the Lora model
# model = PeftModel.from_pretrained(model, peft_model_id)
# # You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference
#
# batch = tokenizer("I really enjoyed the ", return_tensors="pt")
#
# with torch.cuda.amp.autocast():
#     output_tokens = model.generate(**batch, max_new_tokens=50)
#
# print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))