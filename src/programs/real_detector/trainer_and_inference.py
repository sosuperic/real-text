"""


-----
TRAIN
-----
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python src/programs/real_detector/trainer.py \
--data_gen_method gpt2-xl_k40 \
--wandb_group realdetector_realnews \
--model_name bert-base-uncased \
--batch_size 32 --n_gpus 1 \
--lr 1e-5 \
--epochs 10

Batch size 32 with length 192

---------
INFERENCE
---------
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python src/programs/real_detector/trainer.py \
--checkpoint wandb/run-20200803_230036-27j3kdrk/epoch=0.ckpt --eval_out_fp evalETS_v0.1.json \
--batch_size 32 \
--run_eval \
--eval_dataset ets \
--wandb_group realdetect_v0_evalETS

--checkpoint wandb/run-20200803_230036-27j3kdrk/epoch=0.ckpt --eval_out_fp evalETS_v0.1.json \
--checkpoint wandb/run-20200803_212506-27tjk6r7/epoch=1.ckpt --eval_out_fp evalETS_v0.0.json \

----
TODO
---- 
- UserWarning: Did not find hyperparameters at model hparams. Saving checkpoint without hyperparameters.
- MultiGPU

"""

import argparse
from functools import wraps
import glob
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
print(os.path.abspath(os.getcwd()))
import time
from tqdm import tqdm
from typing import Callable

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import wandb


from config import ROOT_PATH
from src.programs.real_detector.model import RealDetector
from src.programs.real_detector.data import get_etsnonnative_dataloader
# Normally, I run using PYTHONPATH=.
# with wandb sweep sweep.yaml, couldn't figure out how to set PYTHONPATH. So change dir in order
# to be able to access data
os.chdir(ROOT_PATH)

def get_trainer_args():

    parser = argparse.ArgumentParser()


    parser.add_argument('--data_gen_method', default='gpt2-xl_p96', type=str, required=False)

    parser.add_argument('--run_eval', default=False, action="store_true", required=False)
    parser.add_argument('--eval_dataset', default=None, type=str, required=False)
    parser.add_argument('--eval_out_fp', default=None, type=str, required=False)

    # parser.add_argument('--seq_len', default=256, type=int, required=False)
    # parser.add_argument('--n_tokens', default=-1, type=int, required=False)
    # parser.add_argument('--n_batches', default=-1, type=int, required=False)
    # parser.add_argument('--fast', default=False,
    #                     action="store_true", required=False)
    # parser.add_argument('--efficient', default=False,
    #                     action="store_true", required=False)

    parser.add_argument('--model_name', default='bert-base-uncased', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--disable_lr_schedule',
                        default=False, action='store_true')

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--grad_steps', default=1, type=int)
    parser.add_argument('--epochs', default=15, type=int)

    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--apex_mode', default='O1', type=str)

    parser.add_argument('--track_grad_norm', default=-1, type=int)
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--debug_run', default=False, action="store_true")

    parser.add_argument('--wandb_notes', default=None, type=str)
    parser.add_argument('--wandb_group', default=None, type=str)

    args = parser.parse_args()
    return args

def setup_wandb():
    wandb.login()
    experiment = wandb.init(project="real-text", reinit=True,
                            notes=args.wandb_notes, group=args.wandb_group)
    wandb_logger = WandbLogger(save_dir=ROOT_PATH, experiment=experiment)
    wandb_logger.log_hyperparams(args)
    return wandb_logger


if __name__ == "__main__":
    args = get_trainer_args()
    wandb_logger = setup_wandb()
    # train
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint_callback = ModelCheckpoint(wandb.run.dir, save_top_k=1)

    
    trainer = pl.Trainer(max_epochs=args.epochs,
                         accumulate_grad_batches=args.grad_steps, 
                         gpus=args.n_gpus,
                         precision=args.precision, amp_level=args.apex_mode,
                         resume_from_checkpoint=args.checkpoint,
                         logger=wandb_logger,
                         track_grad_norm=args.track_grad_norm,
                         fast_dev_run=args.debug_run,
                         early_stop_callback=early_stopping_callback,
                         checkpoint_callback=checkpoint_callback,
                         progress_bar_refresh_rate=1,
                         num_sanity_val_steps=0,
                        #  log_gpu_memory='all',
                         )


    if args.run_eval:
        if args.eval_dataset == 'ets':
            loader = get_etsnonnative_dataloader()
        model = RealDetector.load_from_checkpoint(args.checkpoint, args)
        trainer.test(model=model, test_dataloaders=loader)
    else:
        model = RealDetector(args)
        trainer.fit(model)

