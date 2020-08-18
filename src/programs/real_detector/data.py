"""

Usage: 
    PYTHONPATH=. python src/programs/real_detector/data.py
"""

import json
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from config import REAL_TEXT_PATH, GENERATED_TEXT_PATH, PREPPED_REALGEN_TEXT_PATH, \
    ETS_NONNATIVE_PATH
from src.utils import save_file, load_file


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


##########################################################################
# Training dataset
##########################################################################
class RealGeneratedTextDataset(Dataset):
    def __init__(self, split, gen_method):
        super().__init__()
        self.split = split

        fp = PREPPED_REALGEN_TEXT_PATH / f'{gen_method}_{split}.pkl'
        self.data = load_file(fp)

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

def get_realgentext_dataloader(split, gen_method, **kwargs):
    ds = RealGeneratedTextDataset(split, gen_method)
    loader = DataLoader(ds, **kwargs)
    return loader

##########################################################################
# Inference datasets
##########################################################################
class ETSNonNativeDataset(Dataset):
    def __init__(self):
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = self.load_data(tokenizer)

    def load_data(self, tokenizer, max_len=192):
        print('Loading ETSNonNative data')
        data = []

        fp = ETS_NONNATIVE_PATH / 'index.csv'
        df = pd.read_csv(fp)
        for i, row in df.iterrows():
            fn, prompt, lang, score = row.Filename, row.Prompt, row.Language, row['Score Level']
            fp = ETS_NONNATIVE_PATH / 'responses/original/' / fn
            text = ''.join(open(fp).readlines())

            # tokenize, convert to ids
            text_trunc, text_len, token_ids_padded = tokenize_and_prep(tokenizer, text, max_len)
            if text_len < max_len:
                continue

            data.append({
                'fn': fn, 'fp': fp, 'token_ids_padded': token_ids_padded,
                'text': text_trunc, 'prompt': prompt, 'lang': lang, 'score': score,
            })

            # if i == 25:
            #     break
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        token_ids_padded = torch.LongTensor(item['token_ids_padded'])
        text, id = item['text'], item['fn']
        label = 1  # human-written

        atn_mask = torch.ones(token_ids_padded.size(0))

        return token_ids_padded, atn_mask, label, text, id


def get_etsnonnative_dataloader(**kwargs):
    ds = ETSNonNativeDataset()
    loader = DataLoader(ds, **kwargs)
    return loader


if __name__ == '__main__':
    # prep_and_save_data('gpt2-xl_p96')
    # prep_and_save_data('gpt2-xl_p100')
    prep_and_save_data('gpt2-xl_k40')


    # ds = RealGeneratedTextDataset('valid', 'gpt2-xl_p96')
    # loader = get_realgentext_dataloader('valid', 'gpt2-xl_p96', batch_size=8, shuffle=True)
    # for batch in loader:
    #     breakpoint()
    
    # loader = get_etsnonnative_dataloader(batch_size=4, shuffle=True)
    # for batch in loader:
    #     breakpoint()
