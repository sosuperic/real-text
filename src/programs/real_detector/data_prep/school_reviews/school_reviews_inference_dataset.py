"""
Just load the real school reviews to test
a detector on
"""
import json
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from config import SCHOOL_REVIEWS_DATA

from src.programs.real_detector.dataset_utils import tokenize_and_prep
from src.utils import save_file, load_file

class SchoolReviewsRealTextDataset(Dataset):
    """
    """
    def __init__(self, gen_method=None, max_len=128):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        
        # Load data
        df = pd.read_csv(SCHOOL_REVIEWS_DATA)  # each row is one school
        urls = df.url  # url per school 
        revs = df.review_text.tolist()  # human reviews

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
                'id': urls[i],
            }
            
            self.data.append(d)

            # if i == 10:
            #     break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        token_ids_padded = item['token_ids_padded']
        label = 1  # human-written
        text = item['text_trunc']
        id = item['id']

        text_len = item['text_len']
        seq_len = token_ids_padded.size(0)
        attn_mask = torch.zeros(seq_len)
        for i in range(min(seq_len, text_len)):
            attn_mask[i] = 1

        return token_ids_padded, attn_mask, label, text, id


def get_schoolreviewsreal_dataloader(max_len=128,
    **kwargs):
    ds = SchoolReviewsRealTextDataset(max_len=max_len)
    loader = DataLoader(ds, **kwargs)
    return loader