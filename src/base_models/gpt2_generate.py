
"""
Script to generate text using GPT2Wrapper.

Usage:
    PYTHONPATH=. python src/base_models/gpt2_generate.py
"""

import random
import torch

from src.base_models.gpt2_wrapper import get_base_args, add_generation_args, set_seed_for_gen, \
    GPT2Wrapper


if __name__ == "__main__":
    ### Following currently generates unconditionally, maybe should be moved to a separate file

    parser = get_base_args()
    parser = add_generation_args(parser)
    args = parser.parse_args()
    set_seed_for_gen(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2 = GPT2Wrapper(args)
    gpt2 = gpt2.to(device)


    print('UNCONDITIONAL GENERATION:')
    gpt2.generate_unconditional(n=2, stdout=True)
    print()
    print('=' * 100)
    print('=' * 100)
    print('=' * 100)
    print()
    print('CONDITIONAL GENERATION:')
    gpt2.generate_conditional(prompts=[
        'The mitochondria is the ',
        'Stay on designated trails and roads. '
    ])

