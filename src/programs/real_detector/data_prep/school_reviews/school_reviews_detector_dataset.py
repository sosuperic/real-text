"""
Mix of real and generated school reviews to train detector
"""


import json
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from config import SCHOOL_REVIEWS_DATA, SCHOOL_REVIEWS_TRAIN_DETECTOR_PATH

from src.programs.real_detector.dataset_utils import tokenize_and_prep
from src.utils import save_file, load_file


class SchoolReviewsRealGeneratedTextDataset(Dataset):
    """
    """
    def __init__(self, split, gen_method=None, max_len=192):
        super().__init__()
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        
        # Load data
        df = pd.read_csv(SCHOOL_REVIEWS_DATA)
        human_revs = df.review_text.tolist()
        gen_revs = load_file(SCHOOL_REVIEWS_TRAIN_DETECTOR_PATH)

        # Get data for this split
        if split == 'train':
            human_revs = human_revs[:5000]
            gen_revs = gen_revs[:5000]
        elif split == 'valid':
            human_revs = human_revs[5000:6000]
            gen_revs = gen_revs[5000:6000]
        elif split == 'test':
            human_revs = human_revs[6000:7000]
            gen_revs = gen_revs[6000:7000]
        revs = human_revs + gen_revs

        # Prep data
        self.data = []
        for i, rev in enumerate(revs):
            text_trunc, text_len, token_ids_padded = tokenize_and_prep(
                self.tokenizer, rev, max_len)

            d = {
                'review': rev,
                'text_trunc': text_trunc,
                'text_len': text_len,
                'token_ids_padded': token_ids_padded,
            }
            
            self.data.append(d)

            # if i == 10:
            #     break

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
