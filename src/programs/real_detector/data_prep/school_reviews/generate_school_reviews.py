"""
Load a fine-tuned school-reviews model and generate reviews
to be used for a human/machine classifier.

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python src/data_prep/generate_school_reviews.py
"""

from src.base_models.gpt2_wrapper import get_base_args, add_generation_args, set_seed_for_gen, GPT2Wrapper
from src.utils import save_file, load_file



import torch

if __name__ == '__main__':
    parser = get_base_args()
    parser = add_generation_args(parser)
    args = parser.parse_args()
    set_seed_for_gen(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gpt2-medium trained on one week of covid news
    model_fp = 'trained_models/school_reviews/gpt2/wandb/model_e0.pkl'
    tokenizer_fp = 'trained_models/school_reviews/gpt2/wandb/tokenizer.pkl'
    print('Loading')

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    model, tokenizer = GPT2LMHeadModel, GPT2TokenizerFast
    model = model.from_pretrained('gpt2-xl')

    model = load_file(model_fp)
    tokenizer = load_file(tokenizer_fp)

    gpt2 = GPT2Wrapper(args, model=model, tokenizer=tokenizer)
    gpt2 = gpt2.to(device)
    print('Loaded')

    OUT_FP = 'data/school_reviews/train_detector/trainedonallreviews_gpt2-xl_e0.json'

    texts = []
    for i in range(5000):
        text = model.generate_unconditional(self, n=1, bsz=1, stdout=True)[0]
        texts.append(text)
        if i % 10 == 0:
            save_file(texts, OUT_FP)
