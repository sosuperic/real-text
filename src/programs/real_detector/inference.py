
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
# Normally, I run using PYTHONPATH=.
# with wandb sweep sweep.yaml, couldn't figure out how to set PYTHONPATH. So change dir in order
# to be able to access data
os.chdir(ROOT_PATH)


class ETSDataset():
    

def get_trainer_args():

    parser = argparse.ArgumentParser()


    parser.add_argument('--data_gen_method', default='gpt2-xl_p96', type=str, required=True)
    parser.add_argument('--run_eval', default=False,
                        action="store_true", required=False)

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

if __name__ == "__main__":
    args = get_trainer_args()
    wandb_logger = setup_wandb()
    # train
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint_callback = ModelCheckpoint(wandb.run.dir, save_top_k=1)

    model = RealDetector(args)
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

    trainer.fit(model)