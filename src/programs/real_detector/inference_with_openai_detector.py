
"""
Load released OpenAI detector and run inference on it

----------------
INFERENCE ON ETS
----------------
PYTHONPATH=. python src/programs/real_detector/inference_with_openai_detector.py \
--model_name roberta-base \
--dataset ets \
--output_dir outputs/openai_detector/ets/

---------------------------
INFERENCE ON SCHOOL REVIEWS
---------------------------
PYTHONPATH=. python src/programs/real_detector/inference_with_openai_detector.py \
--model_name roberta-base \
--dataset schoolreviews \
--output_dir outputs/openai_detector/school_reviews/
"""

import argparse
import os
from pathlib import Path
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from config import OPENAI_DETECTOR_DIR
from src.programs.real_detector.data_prep.ets.ets_dataset import get_etsnonnative_dataloader
from src.programs.real_detector.data_prep.school_reviews.school_reviews_inference_dataset import \
    get_schoolreviewsreal_dataloader
from src.utils import save_file


def load_model(args):
    model = RobertaForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    if args.model_name == 'roberta-base':
        checkpoint = OPENAI_DETECTOR_DIR / 'detector-base.pt'
    elif args.model_name == 'roberta-large':
        checkpoint = OPENAI_DETECTOR_DIR / 'detector-large.pt'
    data = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    return model, tokenizer

def inference_on_dataset(args):
    model, tokenizer = load_model(args)
    model.cuda()
    if args.dataset == 'ets':
        data_loader = get_etsnonnative_dataloader(
            max_len=args.max_len,
            batch_size=1)
    elif args.dataset == 'schoolreviews':
        data_loader = get_schoolreviewsreal_dataloader(
            max_len=args.max_len,
            batch_size=1)
    print('Dataset loaded')
    

    results = {}
    for i, batch in enumerate(data_loader):
        if i % 100 == 0:
        print('{} / {} / ')

        token_ids_padded, atn_mask, label, text, id = batch
        query = text[0]  # bsz 1
        id = id[0]  # bsz 1

        tokens = tokenizer.encode(query)
        all_tokens = len(tokens)
        tokens = tokens[:tokenizer.max_len - 2]
        used_tokens = len(tokens)
        tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
        mask = torch.ones_like(tokens)

        with torch.no_grad():
            tokens = tokens.cuda()
            mask = mask.cuda()
            logits = model(tokens, attention_mask=mask)[0]
            probs = logits.softmax(dim=-1)

            fake, real = probs.detach().cpu().flatten().numpy().tolist()

            correct = False
            if ((real > fake) and (label == 1)) or \
                (fake > real) and (label == 0):
                correct = True
            results[id] = {'fake': fake, 'real': real, 'correct': correct}

    save_file(results, os.path.join(args.output_dir, 'results.json'), verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='roberta-base')
    parser.add_argument('--dataset', default='ets')
    parser.add_argument('--max_len', default=128)
    parser.add_argument('--output_dir', default=None, required=True)
    args = parser.parse_args()
    
    inference_on_dataset(args)



