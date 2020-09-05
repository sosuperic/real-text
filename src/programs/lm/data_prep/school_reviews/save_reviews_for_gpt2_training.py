"""
Usage:
    PYTHONPATH=. python src/data_prep/school_reviews.py
"""

import numpy as np
import pandas as pd

from config import SCHOOL_REVIEWS_DATA, SCHOOL_REVIEWS_TRAINGPT2_DIR

def save_comments_for_gpt2_training():
    df = pd.read_csv(SCHOOL_REVIEWS_DATA)
    out_dir = SCHOOL_REVIEWS_TRAINGPT2_DIR

    # get train and valid indices
    all_ind = list(range(0, len(df)))
    np.random.shuffle(all_ind)
    train_frac = 0.9
    train_ind = all_ind[0:int(train_frac*len(all_ind))]
    val_ind = all_ind[int(train_frac*len(all_ind)):]

    # use indices to get text
    train_text =  list(df['review_text'][train_ind])
    valid_text = list(df['review_text'][val_ind])

    # save
    def save_to_file(texts, fp):
        with open(fp, 'w') as f:
            for line in texts:
                print(repr(line), file=f)

    train_fp = out_dir / 'train' / 'train.txt'
    valid_fp = out_dir / 'valid' / 'valid.txt'
    save_to_file(train_text, train_fp)
    save_to_file(valid_text, valid_fp)
    
if __name__ == "__main__":
    save_comments_for_gpt2_training()