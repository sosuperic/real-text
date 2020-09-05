
"""
Generate texts using specific decoding hparams, conditioned on prefixes of
real data.

This is data to be used to 

Usage:
PYTHONPATH=. python src/programs/real_detector/generate_realfake_rct.py

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python src/sandbox/generate_realfake_rct.py -s 0 -e 10000 --split train
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python src/sandbox/generate_realfake_rct.py -s 10000 -e 20000 --split train
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python src/sandbox/generate_realfake_rct.py -s 20000 -e 30000 --split train
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python src/sandbox/generate_realfake_rct.py -s 30000 -e 40000 --split train
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python src/sandbox/generate_realfake_rct.py -s 40000 -e 50000 --split train
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python src/sandbox/generate_realfake_rct.py -s 50000 -e 60000 --split train
"""

import json
import jsonlines
import os
import random
import torch

from src.base_models.gpt2_wrapper import get_base_args, add_generation_args, set_seed_for_gen, \
    GPT2Wrapper

REAL_TRAIN_FP = '/mas/projects/continual-language/sandbox-echu/vendor/gpt-2-output-dataset/data/webtext.train.jsonl'
REAL_VALID_FP = '/mas/projects/continual-language/sandbox-echu/vendor/gpt-2-output-dataset/data/webtext.valid.jsonl'
REAL_TEST_FP = '/mas/projects/continual-language/sandbox-echu/vendor/gpt-2-output-dataset/data/webtext.test.jsonl'


def add_args(parser):
    parser.add_argument('-s', '--start', default=None, type=int)
    parser.add_argument('-e', '--end', default=None, type=int)
    parser.add_argument('--split', default='test', type=str)
    return parser

def get_prefixs(split):
    if split == 'train':
        fp = REAL_TRAIN_FP
    elif split == 'valid':
        fp = REAL_VALID_FP
    elif split == 'test':
        fp = REAL_TEST_FP

    id_prefixs = []
    with open(fp, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            id, text = data['id'], data['text']
            prefix = text.split()[0]
            id_prefixs.append((id, prefix))
    return id_prefixs

def generate(gpt2, split, out_dir, start=None, end=None):
    print('Getting prefixes')
    id_prefixs = get_prefixs(split)
    if (start is not None) and (end is not None):
        id_prefixs = id_prefixs[start:end]
        print(len(id_prefixs))

    # get out path
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_fn = '{}'.format(split)
    if (start is not None) and (end is not None):
        out_fn += '_{}-{}'.format(start, end)
    out_fn += '.jsonl'
    out_fp = os.path.join(out_dir,  out_fn)
    print(out_fp)

    print('Generating text')
    output = []
    step = 10
    writer = jsonlines.Writer(open(out_fp, 'w'), flush=True)
    for i in range(0, len(id_prefixs), step):
        print('GENERATED: ', i)
        cur_ids, cur_prefixs = zip(*id_prefixs[i:i+step])
        cur_gentexts = gpt2.generate_conditional(prompts=cur_prefixs)

        # add to batch
        cur_batch = [{'id': cur_ids[j], 'text': cur_gentexts[j]} for j in range(len(cur_ids))]
        for item in cur_batch:
            writer.write(item)

if __name__ == "__main__":
    ### Following currently generates unconditionally, maybe should be moved to a separate file
    parser = get_base_args()
    parser = add_generation_args(parser)
    parser = add_args(parser)
    args = parser.parse_args()
    set_seed_for_gen(args)

    # SETUP
    # args.model_name = 'distilgpt2'
    args.model_name = 'gpt2-xl'
    # args.top_p = 0.96
    args.top_p = 1.00
    args.sample_len = 192
    # out_dir = 'realfake/generate_realfake_rct-p96'
    out_dir = 'realfake/fake_gpt2-xl_p100'
    gen_str = None
    
    if args.top_p == 0.96 and args.top_k == 0:
        gen_str = 'p96'
    if args.top_p == 1.00 and args.top_k == 0:
        gen_str = 'p100'
    elif args.top_k == 40 and args.top_p == 1.0:
        gen_str = 'k40'
    else:
        print('Bad combo of top-p / top-k')
        import sys
        sys.exit()
    out_dir = 'realfake/fake_{}_{}'.format(args.model_name, gen_str)

    gpt2 = GPT2Wrapper(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2 = gpt2.to(device)

    # GENERATE
    generate(gpt2, args.split, out_dir, start=args.start, end=args.end)