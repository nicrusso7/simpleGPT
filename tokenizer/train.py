from transformers import AutoTokenizer
from datasets import load_dataset, Features


class TokenizerTrainer:

    def __init__(self, model_name, tokenizer_model):
        self._tokenizer_model = tokenizer_model
        self._model_name = model_name

    @staticmethod
    def _get_training_corpus(raw_datasets):
        return (
            raw_datasets["train"][i: i + 100]["text"]
            for i in range(0, len(raw_datasets["train"]), 100)
        )

    @staticmethod
    def test_tokenizer(model_name, input):
        tokenizer = AutoTokenizer.from_pretrained(f"./{model_name}")
        tokens = tokenizer.tokenize(input)
        print(tokens)

    def train(self):
        raw_datasets = load_dataset("json", data_dir="../datasets")
        raw_datasets = raw_datasets.remove_columns("metadata")
        print(raw_datasets['train'])
        base_tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_model)
        # TODO use _get_training_corpus --> https://huggingface.co/course/chapter6/2?fw=pt#assembling-a-corpus
        tokenizer = base_tokenizer.train_new_from_iterator(raw_datasets['train']['text'], 52000)
        tokenizer.save_pretrained(f"../tokenizer/{self._model_name}")


if __name__ == "__main__":
    model = "BocconiGPT_micro_125M"
    trainer = TokenizerTrainer(model, "gpt2")
    trainer.train()
