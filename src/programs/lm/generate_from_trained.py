"""
Load trained Lightning module and generate

Usage:
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python src/programs/lm/generate_from_trained.py
"""

from src.base_models.gpt2_wrapper import get_base_args, add_generation_args, set_seed_for_gen, GPT2Wrapper
from src.programs.lm.lm_trainer import LM

import torch

if __name__ == '__main__':
    parser = get_base_args()
    parser = add_generation_args(parser)
    args = parser.parse_args()
    set_seed_for_gen(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gpt2-medium trained on one week of covid news
    model_fp = '/mas/projects/continual-language/wandb/run-20200714_013537-3oue22mz/epoch=5.ckpt'
    print('Loading model')
    lm_lightning = LM.load_from_checkpoint(model_fp)

    model, tokenizer = lm_lightning.model, lm_lightning.tokenizer
    gpt2 = GPT2Wrapper(args, model=model, tokenizer=tokenizer)
    gpt2 = gpt2.to(device)
    
    print('Generating')
    gpt2.generate_conditional(prompts=[
        'Coronavirus',
        'Covid-19',
        'People who wear masks',
        'The CDC advises',
        'The effects of coronavirus on the upcoming election',
    ])

