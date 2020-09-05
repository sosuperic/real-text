"""
Until we do something more complex, this simply wraps around the
huggingface GPT2 models and adds additional functionalities
(e.g. inference, text generation).

To train a gpt2 model within an Lightning module, we can use the 
huggingface model directly.
"""

import argparse
import random
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, CTRLLMHeadModel, GPT2TokenizerFast, CTRLTokenizer

def get_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--model_name', default='distilgpt2', type=str,
                        help='huggingface modelname, i.e. gpt2-xl, distilgpt2')
    return parser

def add_generation_args(parser):
    # deactivate top_k sampling and sample only from 92% most likely words
    parser.add_argument('--sample_len', default=256, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--top_p', default=0.92, type=float)
    parser.add_argument('--repetition_penalty', default=None, type=float)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--do_sample', default=True, type=bool)
    parser.add_argument('--gen_manual_seed', default=1, type=int)
    parser.add_argument('--gen_manual_seed_bool', default=True, type=int)
    parser.add_argument('--gen_manual_seed_top', default=10000, type=int)
    parser.add_argument('--gen_manual_seed_top_bool', default=True, type=int)
    return parser

def set_seed_for_gen(args):
    # set seed to reproduce results. Feel free to change the seed though to get different results
    actual_seed = None
    if args.gen_manual_seed:
        actual_seed = args.gen_manual_seed_bool
        torch.manual_seed(actual_seed)
    if args.gen_manual_seed_top_bool:
        actual_seed = random.randint(0, args.gen_manual_seed_top)
        torch.manual_seed(actual_seed)

class GPT2Wrapper(nn.Module):
    def __init__(self, args, model=None, tokenizer=None):
        super().__init__()
        self.args = args

        self.model = model
        if model is None:
            model = GPT2LMHeadModel
            print('Loading: ', self.args.model_name)
            self.model = model.from_pretrained(self.args.model_name)

        self.tokenizer = tokenizer
        if tokenizer is None:
            tokenizer = GPT2TokenizerFast
            self.tokenizer = tokenizer.from_pretrained(self.args.model_name)

    def generate_unconditional(self, n=1, bsz=1, stdout=True):
        prompts=["<|endoftext|> " for _ in range(n)]
        return self.generate_conditional(prompts, stdout)

    def generate_conditional(self, prompts, bsz=1, stdout=True):
        """
        TODO: batch generation following https://github.com/huggingface/transformers/issues/3021
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # iterate in batches
        gen_texts = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            prompt = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(device)

            print('Sampling outputs')
            sample_outputs = self.model.generate(input_ids=prompt, do_sample=self.args.do_sample,
                                        top_k=self.args.top_k, top_p=self.args.top_p,
                                        max_length=self.args.sample_len, temperature=self.args.temperature, 
                                        repetition_penalty=self.args.repetition_penalty,
                                        num_return_sequences=self.args.num_return_sequences)
            

            print(len(sample_outputs))
            for i, sample_output in enumerate(sample_outputs):
                gen_text = self.tokenizer.decode(sample_output.cpu().numpy(),
                                                    skip_special_tokens=True)
                gen_texts.append(gen_text)
                if stdout:
                    print("{}/{}: {}".format(i+1, self.args.num_return_sequences, gen_text))

        return gen_texts