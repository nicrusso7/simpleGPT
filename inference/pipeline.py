import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class InferencePipeline:

    @staticmethod
    def run_inference():
        tokenizer = AutoTokenizer.from_pretrained("../models/BocconiGPT_125M/checkpoint-41000")
        model = AutoModelForCausalLM.from_pretrained("../models/BocconiGPT_125M/checkpoint-41000",
                                                     device_map="auto",
                                                     # load_in_8bit=True
                                                     )

        # infer
        # inputs = tokenizer("<human>: I've lost my Bocconi credentials. How can i recover them?\n<bot>:", return_tensors='pt').to(model.device)
        # sample_outputs = model.generate(
        #     **inputs,
        #     # do_sample=True,
        #     # max_length=165,
        #     # penalty_alpha=0.6, top_k=60,
        #     # max_new_tokens=165,
        #     # min_length=50,
        #     max_length=100,
        #     # top_k=50,
        #     temperature=0.1,
        #     # top_p=0.95,
        #     # num_return_sequences=3
        # )
        #
        # print("Output:\n" + 100 * '-')
        # for i, sample_output in enumerate(sample_outputs):
        #     print("{}: {}".format(i, tokenizer.decode(sample_output)))

        # inputs = tokenizer("<human>: who am I speaking to?\n<bot>:", return_tensors='pt')
        # outputs = model.generate(**inputs, max_new_tokens=133, min_length=10, do_sample=False, temperature=0.1, output_scores=True, return_dict_in_generate=True,)
        # transition_scores = model.compute_transition_scores(
        #     outputs.sequences, outputs.scores, normalize_logits=True
        # )
        # input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        # generated_tokens = outputs.sequences[:, input_length:]
        # for tok, score in zip(generated_tokens[0], transition_scores[0]):
        #     # | token | token string | logits | probability
        #     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
        # output_str = tokenizer.decode(outputs[0])
        # print(output_str)
        inputs = tokenizer("<human>: who am I speaking to?\n<bot>:",
                           return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=33, do_sample=False, temperature=0.1)
        output_str = tokenizer.decode(outputs[0])
        print(output_str)
        inputs = tokenizer("<human>: is there an official Bocconi App?\n<bot>:", return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=25, do_sample=False, temperature=0.1)
        output_str = tokenizer.decode(outputs[0])
        print(output_str)
        inputs = tokenizer("<human>: I've lost my Bocconi credentials. How can i recover them?\n<bot>:",
                           return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=28, do_sample=False, temperature=0.1)
        output_str = tokenizer.decode(outputs[0])
        print(output_str)
        inputs = tokenizer("<human>: tell me something about Bocconi university\n<bot>:", return_tensors='pt').to(
            model.device)
        outputs = model.generate(**inputs, max_new_tokens=140, do_sample=False, temperature=0.1)
        output_str = tokenizer.decode(outputs[0])
        print(output_str)
        inputs = tokenizer("<human>: tell me the Bocconi history\n<bot>:", return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=165, do_sample=False, temperature=0.1)
        output_str = tokenizer.decode(outputs[0])
        print(output_str)
        inputs = tokenizer("<human>: I want to visit the campus\n<bot>:", return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=0.1)
        output_str = tokenizer.decode(outputs[0])
        print(output_str)
        inputs = tokenizer("<human>: what are the bachelor fees at bocconi?\n<bot>:", return_tensors='pt').to(
            model.device)
        outputs = model.generate(**inputs, max_new_tokens=42, do_sample=False, temperature=0.1)
        output_str = tokenizer.decode(outputs[0])
        print(output_str)
        inputs = tokenizer("<human>: what are the master fees?\n<bot>:", return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=39, do_sample=False, temperature=0.1)
        output_str = tokenizer.decode(outputs[0])
        print(output_str)
        inputs = tokenizer("<human>: when the semester starts?\n<bot>:", return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=39, do_sample=False, temperature=0.1)
        output_str = tokenizer.decode(outputs[0])
        print(output_str)


if __name__ == "__main__":
    inf_pipe = InferencePipeline()
    inf_pipe.run_inference()




