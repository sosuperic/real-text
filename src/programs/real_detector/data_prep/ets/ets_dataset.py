import json
import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from config import ETS_NONNATIVE_PATH
from src.utils import save_file, load_file
from src.programs.real_detector.dataset_utils import tokenize_and_prep


class ETSNonNativeDataset(Dataset):
    def __init__(self, skip_tokenization=False, max_len=128):
        super().__init__()
        self.skip_tokenization = skip_tokenization
        self.max_len = max_len
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = self.load_data(tokenizer)

    def load_data(self, tokenizer, max_len=128):
        print('Loading ETSNonNative data')
        data = []

        fp = ETS_NONNATIVE_PATH / 'index.csv'
        df = pd.read_csv(fp)
        for i, row in df.iterrows():
            fn, prompt, lang, score = row.Filename, row.Prompt, row.Language, row['Score Level']
            fp = ETS_NONNATIVE_PATH / 'responses/original/' / fn
            text = ''.join(open(fp).readlines())

            # tokenize, convert to ids
            text_trunc, text_len, token_ids_padded = text, None, None
            if not self.skip_tokenization:
                text_trunc, text_len, token_ids_padded = tokenize_and_prep(tokenizer, text, max_len)
                if text_len < max_len:
                    continue

            data.append({
                'fn': fn, 'fp': fp, 'token_ids_padded': token_ids_padded,
                'text': text_trunc, 'prompt': prompt, 'lang': lang, 'score': score,
            })

            # if i == 10:
            #     break

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # dummy to be able to collate together in case skip_tokenization = False
        token_ids_padded = torch.ones(1)  # dummy
        atn_mask = torch.ones(1)  # dummy
        if item['token_ids_padded'] is not None:  # i.e. skip_tokenization=False
            token_ids_padded = torch.LongTensor(item['token_ids_padded'])
            atn_mask = torch.ones(token_ids_padded.size(0))

        text, id = item['text'], item['fn']
        label = 1  # human-written
    
        return token_ids_padded, atn_mask, label, text, id


def get_etsnonnative_dataloader(skip_tokenization=False, max_len=128,
    **kwargs):
    ds = ETSNonNativeDataset(skip_tokenization=skip_tokenization, max_len=max_len)
    loader = DataLoader(ds, **kwargs)
    return loader