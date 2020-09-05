"""
Training datasets so far:
- GPT2
- RealNews (data was already prepped)
- School reviews

Inference datasets:
- ETS

Usage: 
    PYTHONPATH=. python src/programs/real_detector/data.py
"""

import json
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


from src.utils import save_file, load_file



##########################################################################
# Utils
##########################################################################

def tokenize_and_prep(tokenizer, text, max_len):
    tokenized_text = tokenizer.tokenize(text)
    text_len = len(tokenized_text)

    # truncate
    tokenized_text = tokenized_text[:max_len]
    text_trunc = tokenizer.convert_tokens_to_string(tokenized_text)  # return this?

    # add special tokens
    tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)  # returns a list
    token_ids = torch.LongTensor(token_ids)

    token_ids_padded = torch.zeros(max_len + 2).long()   # +2 for special tokens
    token_ids_padded[:len(token_ids)] = token_ids

    return text_trunc, text_len, token_ids_padded


def get_realgentext_dataloader(dataset_name, split, gen_method, **kwargs):
    if dataset_name == 'gpt2':
        from src.programs.real_detector.data_prep.gpt2.gpt2_dataset import GPT2RealGeneratedTextDataset as ds
    elif dataset_name == 'realnews':
        from src.programs.real_detector.data_prep.gpt2.gpt2_dataset import RealNewsRealGeneratedTextDataset as ds
    elif dataset_name == 'school_reviews':
        from src.programs.real_detector.data_prep.school_reviews.school_reviews_detector_dataset import \
            SchoolReviewsRealGeneratedTextDataset

    ds = RealGeneratedTextDataset(split, gen_method)
    loader = DataLoader(ds, **kwargs)
    return loader