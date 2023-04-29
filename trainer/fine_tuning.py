import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, \
    DataCollatorForLanguageModeling

os.environ["WANDB_DISABLED"] = "true"
# load datasets
raw_datasets = load_dataset("json", data_dir="../datasets")
raw_datasets = raw_datasets.remove_columns("metadata")
# create validation set
raw_datasets_clean = raw_datasets["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
raw_datasets_clean["valid"] = raw_datasets_clean.pop("test")
# Add the "test" set to our `DatasetDict`
# raw_datasets_clean["test"] = raw_datasets["test"]
print("Dataset:")
print(raw_datasets_clean)

# print(raw_datasets_clean['train'][:100])
# for row in raw_datasets_clean['train']:
#     print(row)


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"../tokenizer/ChatBase_125M")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_datasets = raw_datasets_clean.map(tokenize_function, batched=True)

# print(tokenizer.decode(tokenized_datasets["train"][10]["input_ids"]))

model = AutoModelForCausalLM.from_pretrained("../models/ChatBase_125M")

training_args = TrainingArguments(output_dir="../models/BocconiGPT_125M",
                                  evaluation_strategy="epoch",
                                  num_train_epochs=1
                                  )

tokenizer.pad_token = tokenizer.eos_token

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    # compute_metrics=compute_metrics,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

trainer.save_model()
