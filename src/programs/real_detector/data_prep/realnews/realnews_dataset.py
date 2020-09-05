"""
Real News dataset was already prepped (i.e. cleaned, split into
train/valid/test) by the peopel who released Grover. This reads and prepares that
"""


import json
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from config import REALNEWS_PREPPED_PATH

from src.programs.real_detector.dataset_utils import tokenize_and_prep
from src.utils import save_file, load_file


class RealNewsRealGeneratedTextDataset(Dataset):
    """
    This is on regular RealNews / Grover

    10K train, 3K val, 12k test

    domain, orig_split, title, url, random_score, ind30k (index) split, 
    authors, date, article, label (human/machine)
    """
    def __init__(self, split, gen_method=None, max_len=192):
        super().__init__()
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Get data for this split
        data = [json.loads(line.strip()) for line in open(REALNEWS_PREPPED_PATH, 'r').readlines()]
        split = 'val' if split == 'valid' else split
        data = [d for d in data if d['split'] == split]
        # then process
        self.data = []
        # i = 0
        for d in data:
            text_trunc, text_len, token_ids_padded = tokenize_and_prep(
                self.tokenizer, d['article'], max_len)
            d['text_trunc'] = text_trunc
            d['text_len'] = text_len
            d['token_ids_padded'] = token_ids_padded
            self.data.append(d)
            # if i == 10:
            #     break
            # i += 1
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        token_ids_padded, label = item['token_ids_padded'], item['label']
        label = 1 if label == 'human' else 0
        text = item['text_trunc']

        # All one's because only sequences at least max_length were generated / selected
        # during the data prep process
        text_len = item['text_len']
        seq_len = token_ids_padded.size(0)
        attn_mask = torch.zeros(seq_len)
        for i in range(min(seq_len, text_len)):
            attn_mask[i] = 1

        return token_ids_padded, attn_mask, label, text

