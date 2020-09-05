

"""
Dataset for training detector or GPT2-generated text

"""

import json
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from config import PREPPED_REALGEN_TEXT_PATH

from src.utils import save_file, load_file
from src.programs.real_detector.dataset_utils import tokenize_and_prep

class GPT2RealGeneratedTextDataset(Dataset):
    """
    This is on regular GPT-2, not RealNews
    """
    def __init__(self, split, gen_method):
        super().__init__()
        self.split = split

        fp = PREPPED_REALGEN_TEXT_PATH / f'{gen_method}_{split}.pkl'
        self.data = load_file(fp)

        if split == 'train':
            import random
            random.shuffle(self.data)
            self.data = self.data[:5000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        token_ids_padded, label = item['token_ids_padded'], item['label']
        text, id = item['text'], item['id']

        # All one's because only sequences at least max_length were generated / selected
        # during the data prep process
        atn_mask = torch.ones(token_ids_padded.size(0))

        return token_ids_padded, atn_mask, label, text
