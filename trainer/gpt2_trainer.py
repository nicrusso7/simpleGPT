import os

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset


class ModelTrainer:

    def __init__(self, model_name):
        self._model_name = model_name

    def train(self):
        os.environ["WANDB_DISABLED"] = "true"
        context_length = 64
        # load datasets
        raw_datasets = load_dataset("json", data_dir="datasets/base-chat/OIG/")
        raw_datasets = raw_datasets.remove_columns("metadata")
        print(raw_datasets)
        # create validation set
        raw_datasets_clean = raw_datasets["train"].train_test_split(train_size=0.85, seed=42)
        # Rename the default "test" split to "validation"
        raw_datasets_clean["valid"] = raw_datasets_clean.pop("test")
        # Add the "test" set to our `DatasetDict`
        # raw_datasets_clean["test"] = raw_datasets["test"]

        print(raw_datasets_clean)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(f"../tokenizer/{self._model_name}")
        # outputs = tokenizer(
        #     raw_datasets["train"]["text"],
        #     truncation=True,
        #     max_length=context_length,
        #     return_overflowing_tokens=True,
        #     return_length=True,
        # )
        #
        # print(f"Input IDs length: {len(outputs['input_ids'])}")
        # print(f"Input chunk lengths: {(outputs['length'])}")
        # print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

        def tokenize(element):
            return tokenizer(
                element["text"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
            # input_batch = []
            # for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            #     if length <= context_length:
            #         input_batch.append(input_ids)
            # return {"input_ids": input_batch}

        tokenized_datasets = raw_datasets_clean.map(
            tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
        )

        print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
        print(tokenizer.decode(tokenized_datasets["train"][1]["input_ids"]))

        print(f"Tokenized Dataset: {tokenized_datasets}")

        # create config
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_ctx=context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # create empty model
        model = GPT2LMHeadModel(config)
        model_size = sum(t.numel() for t in model.parameters())
        print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")

        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        # out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
        # for key in out:
        #     print(f"{key} shape: {out[key].shape}")

        args = TrainingArguments(
            output_dir=f"../models/{self._model_name}/",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="steps",
            eval_steps=3_00,
            logging_steps=5_000,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            weight_decay=0.1,
            warmup_steps=1_000,
            lr_scheduler_type="cosine",
            learning_rate=5e-4,
            save_steps=1_000,
            fp16=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["valid"],
        )

        trainer.train()

        trainer.save_model()


if __name__ == "__main__":
    model = "ChatBase_125M"
    trainer = ModelTrainer(model)
    trainer.train()
