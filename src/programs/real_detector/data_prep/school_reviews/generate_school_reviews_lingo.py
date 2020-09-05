"""
Load trained Lightning module and generate

Usage:
PYTORCH_TRANSFORMERS_CACHE=/raid/lingo/echu/transformers_cache/ \
WANDB_CONFIG_DIR=/raid/lingo/echu/projects/real-text \
CUDA_VISIBLE_DEVICES=13 \
    PYTHONPATH=. python src/data_prep/generate_school_reviews_lingo.py
"""

from src.base_models.gpt2_wrapper import get_base_args, add_generation_args, set_seed_for_gen, GPT2Wrapper
from src.programs.lm.lm_trainer import LM

from src.utils import save_file

import torch

if __name__ == '__main__':
    parser = get_base_args()
    parser = add_generation_args(parser)
    args = parser.parse_args()
    set_seed_for_gen(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.top_p = 0.96


    # get first words of reviews
    import pandas as pd
    df = pd.read_csv('/raid/lingo/echu/projects/data/school_reviews/Parent_gs_comments_by_school_with_covars.csv')
    revs = df.review_text.tolist()
    first_words = [r.split()[0] for r in revs]


    # load model
    model_fp = '/raid/lingo/echu/projects/real-text/trained_models/school_reviews/gpt2/wandb/run-20200901_225031-30fa16p7/epoch=1.ckpt'
    print('Loading model')
    lm_lightning = LM.load_from_checkpoint(model_fp)

    model, tokenizer = lm_lightning.model, lm_lightning.tokenizer
    gpt2 = GPT2Wrapper(args, model=model, tokenizer=tokenizer)
    gpt2 = gpt2.to(device)


    print('Generating')
    # OUT_FP = '../data/school_reviews/train_detector/trainedonallreviews_gpt2-xl_e0_p96.json'
    OUT_FP = '../data/school_reviews/train_detector/trainedonallreviews_gpt2-xl_e1_p96_condgen.json'

    texts = []
    for i in range(7000):
        # text = gpt2.generate_unconditional(n=1, bsz=1, stdout=True)[0]
        text = gpt2.generate_conditional([first_words[i]], bsz=1, stdout=True)[0]
        texts.append(text)
        if i % 10 == 0:
            save_file(texts, OUT_FP)
