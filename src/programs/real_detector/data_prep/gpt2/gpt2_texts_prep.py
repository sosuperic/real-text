"""
Generate text 
"""

import json
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from config import REAL_TEXT_PATH, GENERATED_TEXT_PATH, PREPPED_REALGEN_TEXT_PATH

from src.utils import save_file, load_file




def prep_and_save_data(gen_method='gpt2-xl_p96', max_len=192):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for split in ['test', 'valid', 'train']:
        split_data = []

        # load real
        fp = REAL_TEXT_PATH / f'webtext.{split}.jsonl'
        for line in open(fp):
            item = json.loads(line)
            id, ended, text = item['id'], item['ended'], item['text']

            text_trunc, text_len, token_ids_padded = tokenize_and_prep(tokenizer, text, max_len)
            if text_len < max_len:
                continue
            prepped = {
                'id': id,
                'text': text_trunc,
                'token_ids_padded': token_ids_padded,
                'label': 1,
                'label_text': 'human-written'
            }
            split_data.append(prepped)
        
        # load fake
        dir = GENERATED_TEXT_PATH / f'fake_{gen_method}'
        for fn in os.listdir(dir):
            if fn.startswith(split):
                fp = os.path.join(dir, fn)
                for line in open(fp):
                    item = json.loads(line)
                    id, text = item['id'], item['text']

                    text_trunc, text_len, token_ids_padded = tokenize_and_prep(tokenizer, text, max_len)
                    if text_len < max_len:
                        continue
                    prepped = {
                        'id': id,
                        'text': text_trunc, 
                        'token_ids_padded': token_ids_padded,
                        'label': 0,
                        'label_text': 'machine-generated'
                    }
                    split_data.append(prepped)

        out_fp = PREPPED_REALGEN_TEXT_PATH / f'{gen_method}_{split}.pkl'
        save_file(split_data, out_fp, verbose=True)