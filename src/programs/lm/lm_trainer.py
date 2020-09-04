"""
Train a language model

Usage:

Train on school reviews:
WANDB_DIR=/u/echu/projects/research/real-text/trained_models/school_reviews/gpt2 \
WANDB_PROJECT=real-text-school-reviews \
WANDB_NAME=gpt2-small-TESTING-LINGO \
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python src/programs/lm/lm_trainer.py \
--train_path /u/echu/projects/research/real-text/data/school_reviews/train_gpt2/train/ \
--val_path /u/echu/projects/research/real-text/data/school_reviews/train_gpt2/valid/ \
--test_path /u/echu/projects/research/real-text/data/school_reviews/train_gpt2/valid \
--model_name gpt2-small --batch_size 1 \
--epochs 10 \
--wandb_project real-text-school-reviews \
--wandb_group all_reviews_no_strat

gpt2-medium takes 8GB to train

--wandb_savedir trained_models/school_reviews/gpt2 # this doesn't seem to do anything


##################################

Train on linksearch data:

Test run as follows, but run sweep using sweep.yaml

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python src/programs/lm/lm_trainer.py \
--train_path /mas/projects/continual-language/data/news/lsm_tweet_links/linksearch/parse_scrape_test/parsed/RS_2020-07_lm-dataset/train/ \
--val_path /mas/projects/continual-language/data/news/lsm_tweet_links/linksearch/parse_scrape_test/parsed/RS_2020-07_lm-dataset/val/ \
--test_path /mas/projects/continual-language/data/news/lsm_tweet_links/linksearch/parse_scrape_test/parsed/RS_2020-07_lm-dataset/test/ \
--wandb_group articles_v0 \
--model_name gpt2-medium --batch_size 2 \
--epochs 10

With batch size 4, gpt2-medium takes about 10GB.

Train on school reviews data:


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
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, CTRLLMHeadModel, GPT2TokenizerFast, CTRLTokenizer, AdamW, get_linear_schedule_with_warmup
import wandb


# from config import ROOT_PATH
# Normally, I run using PYTHONPATH=.
# with wandb sweep sweep.yaml, couldn't figure out how to set PYTHONPATH. So change dir in order
# to be able to access data
# os.chdir(ROOT_PATH)


MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2TokenizerFast),
    # 'ctrl': (CTRLLMHeadModel, CTRLTokenizer)
}


######################################################################################################
#
# Dataset
#
######################################################################################################
class TextDataset(torch.utils.data.Dataset):
    """
    Path is directory with txt files
    """
    def __init__(self, path, tokenizer, args):
        start = time.time()

        self.n_original_tokens = 0
        self.n_tokens = 0

        print('About to create batches')

        # Create batches
        if os.path.isdir(path):
            self.batches = []
            for f in glob.glob(os.path.join(path, '*.txt')):
                self.batches += self._tokenize(f, tokenizer, args)
        else:
            self.batches = self._tokenize(path, tokenizer, args)

        end = time.time()

        print(f'Dataset created in {int(end - start)} seconds')
        print(f'Dataset length: {len(self.batches)}')
        print(
            f'Num tokens: {self.n_tokens} | Num original tokens: {self.n_original_tokens}')

    def _tokenize(self, path, tokenizer, args):
        batches = []
        text = []
        with open(path, encoding="utf-8") as handle:
            # efficient uses less memory by going line-by-line. Drawbacks: if len(line) isn't a multiple of seq_len, the remainder will be left
            if args.efficient or args.fast:
                for line in handle:
                    self.n_original_tokens += len(line.split(" "))
                    if len(line) > 0 and not line.isspace():
                        text.append(line)
            # Default way reads in entire file into memory
            else:
                temp = handle.read()
                text.append(temp)
                self.n_original_tokens += len(temp.strip().split(" "))

        # Fast way uses `batch_encode_plus`. Drawbacks: only the first seq_len chars get kept
        if args.fast:
            batches = tokenizer.batch_encode_plus(
                text, add_special_tokens=True, max_length=args.seq_len)["input_ids"]
        else:
            for l in tqdm(text):
                tokenized_text = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(l))

                if args.n_tokens > -1:
                    tokenized_text = tokenized_text[:args.n_tokens]

                if len(tokenized_text) < args.seq_len:
                    batches.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text))
                else:
                    for i in range(len(tokenized_text) // args.seq_len):
                        batches.append(tokenizer.build_inputs_with_special_tokens(
                            tokenized_text[i * args.seq_len: (i + 1) * args.seq_len]))

                if args.n_batches > -1 and len(batches) >= args.n_batches:
                    break

        self.n_tokens += sum([len(batch) for batch in batches])

        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return torch.tensor(self.batches[index])

######################################################################################################
#
# Trainer
#
######################################################################################################
class LM(pl.LightningModule):
    def __init__(self, hparams):
        super(LM, self).__init__()
        self.hparams = hparams
        self.args = hparams

        model, tokenizer = MODEL_CLASSES[self.args.model_type]
        self.model = model.from_pretrained(self.args.model_name)
        self.tokenizer = tokenizer.from_pretrained(self.args.model_name)

        print(self.args.train_path)
        self.train_dataset = TextDataset(
            self.args.train_path, self.tokenizer, self.args)

        self.table_data = []

    def forward(self, inputs, labels):
        return self.model(inputs, labels=labels)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, batch)[0]

        if self.args.disable_lr_schedule:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
        else:
            lr = self.scheduler.get_last_lr()[0]

        return {'loss': loss, "log": {"train_loss": loss.item(), "learning_rate": lr}}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, batch)[0]

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_ppl = torch.exp(val_loss_mean)
        adjusted_val_loss = val_loss_mean * \
            ((self.val_dataset.n_tokens - 1) /
             (self.val_dataset.n_original_tokens - 1))
        adjusted_val_ppl = torch.exp(adjusted_val_loss)

        if self.args.accelerator != "TPU":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            prompt = torch.tensor(self.tokenizer.encode("<|endoftext|> ")).unsqueeze(0).to(device)
            outputs = self.model.generate(input_ids=prompt, max_length=self.args.sample_len, temperature=self.args.temperature,
                                          top_k=self.args.top_k, top_p=self.args.top_p, repetition_penalty=self.args.repetition_penalty, num_return_sequences=1)
            outputs = self.tokenizer.decode(outputs[0].cpu().numpy(), skip_special_tokens=True)
            print("\nSampling:")
            print(outputs)
            print("\n")

            self.table_data.append([f'{self.trainer.current_epoch}', outputs])

        metrics = {'val_loss': val_loss_mean, 'val_ppl': val_ppl, 'adjusted_val_ppl': adjusted_val_ppl, "log": {
            'val_loss': val_loss_mean, 'val_ppl': val_ppl, 'adjusted_val_ppl': adjusted_val_ppl, "samples": wandb.Table(columns=['Epoch', 'Text'], data=self.table_data)}}

        return metrics

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch, batch)[0]

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_ppl = torch.exp(test_loss_mean)
        adjusted_test_ppl = torch.exp(
            test_loss_mean * ((self.test_dataset.n_tokens - 1) / (self.test_dataset.n_original_tokens - 1)))

        if args.accelerator != "TPU":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            prompt = torch.tensor(self.tokenizer.encode("<|endoftext|> ")).unsqueeze(0).to(device)
            outputs = self.model.generate(input_ids=prompt, max_length=self.args.sample_len, temperature=self.args.temperature,
                                          top_k=self.args.top_k, top_p=self.args.top_p, repetition_penalty=self.args.repetition_penalty, num_return_sequences=1)
            outputs = self.tokenizer.decode(outputs[0].cpu().numpy(), skip_special_tokens=True)
            print("Sampling:")
            print(outputs)
            print("\n")

            self.table_data.append([f'{self.trainer.current_epoch}', outputs])

        metrics = {'test_epoch_loss': test_loss_mean,
                   'test_ppl': test_ppl, 'adjusted_test_ppl': adjusted_test_ppl, "log": {'test_epoch_loss': test_loss_mean, 'test_ppl': test_ppl, 'adjusted_test_ppl': adjusted_test_ppl, "samples": wandb.Table(columns=['Epoch', 'Text'], data=self.table_data)}}

        return metrics

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if args.optimizer == 'AdamW':
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.lr, eps=1e-8)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters, lr=args.lr)

        if self.args.disable_lr_schedule:
            return optimizer
        else:
            n_accelerators = 1
            if self.args.n_tpu_cores != None:
                n_accelerators *= self.args.n_tpu_cores
            elif self.args.n_gpus != None:
                n_accelerators *= self.args.n_gpus

            train_steps = int(
                (len(self.train_dataset) / (self.args.batch_size * n_accelerators)) * self.args.epochs)

            self.scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1 * train_steps), num_training_steps=train_steps)

            return [optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

    def collate(self, examples):
        if self.tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def train_dataloader(self):
        if self.args.accelerator == "TPU":
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )
        else:
            sampler = torch.utils.data.RandomSampler(self.train_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, num_workers=4, collate_fn=self.collate, sampler=sampler)

        return train_dataloader

    def val_dataloader(self):
        self.val_dataset = TextDataset(
            self.args.val_path, self.tokenizer, self.args)

        sampler = None
        if self.args.accelerator == "TPU":
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )

        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, num_workers=4, collate_fn=self.collate, sampler=sampler, shuffle=False)

        return val_dataloader

    def test_dataloader(self):
        self.test_dataset = TextDataset(
            self.args.test_path, self.tokenizer, self.args)

        sampler = None
        if self.args.accelerator == "TPU":
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )

        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.batch_size, num_workers=4, collate_fn=self.collate, sampler=sampler, shuffle=False)

        return test_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', default=None, type=str, required=False)
    parser.add_argument('--val_path', default=None, type=str, required=False)
    parser.add_argument('--test_path', default=None, type=str, required=False)
    parser.add_argument('--run_eval', default=False,
                        action="store_true", required=False)

    parser.add_argument('--seq_len', default=256, type=int, required=False)
    parser.add_argument('--n_tokens', default=-1, type=int, required=False)
    parser.add_argument('--n_batches', default=-1, type=int, required=False)
    parser.add_argument('--fast', default=False,
                        action="store_true", required=False)
    parser.add_argument('--efficient', default=False,
                        action="store_true", required=False)

    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--model_name', default='distilgpt2', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--disable_lr_schedule',
                        default=False, action='store_true')

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--grad_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--accelerator', default='GPU', type=str)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--n_tpu_cores', default=None, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--apex_mode', default='O1', type=str)

    parser.add_argument('--sample_len', default=256, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_k', default=None, type=int)
    parser.add_argument('--top_p', default=0.96, type=float)
    parser.add_argument('--repetition_penalty', default=None, type=float)

    parser.add_argument('--track_grad_norm', default=-1, type=int)

    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--debug_run', default=False, action="store_true")

    parser.add_argument('--wandb_group', default=None, type=str)
    parser.add_argument('--wandb_savedir', default=None, type=str)
    parser.add_argument('--wandb_notes', default=None, type=str)
    parser.add_argument('--wandb_project', default=None, type=str)

    args = parser.parse_args()

    if args.accelerator == "TPU":
        args.global_batch_size = args.batch_size * args.n_tpu_cores

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    if args.accelerator == 'TPU':
        import torch_xla.core.xla_model as xm

    # setup wandb
    wandb.login()
    experiment = wandb.init(project=args.wandb_project, reinit=True,
                            notes=args.wandb_notes, group=args.wandb_group)
    wandb_logger = WandbLogger(save_dir=args.wandb_savedir, experiment=experiment)
    wandb_logger.log_hyperparams(args)

    # train
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint_callback = ModelCheckpoint(wandb.run.dir, save_top_k=1)

    model = LM(args)
    trainer = pl.Trainer(max_epochs=args.epochs,
                         accumulate_grad_batches=args.grad_steps, 
                         gpus=args.n_gpus, 
                        #  num_tpu_cores=args.n_tpu_cores,
                         precision=args.precision, amp_level=args.apex_mode,
                         resume_from_checkpoint=args.checkpoint,
                         logger=wandb_logger,
                         track_grad_norm=args.track_grad_norm,
                         fast_dev_run=args.debug_run,
                         early_stop_callback=early_stopping_callback,
                         checkpoint_callback=checkpoint_callback,
                         progress_bar_refresh_rate=1,
                         num_sanity_val_steps=0,
                         log_gpu_memory='all',
                         )

    trainer.fit(model)