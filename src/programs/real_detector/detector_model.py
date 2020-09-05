
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from src.utils import save_file
from src.programs.real_detector.data import get_realgentext_dataloader

class RealDetector(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args
        
        # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForSequenceClassification
        self.model = BertForSequenceClassification.from_pretrained(
            self.args.model_name,
            num_labels=2,
            output_attentions=False
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # attention would be: (batch_size, num_heads, sequence_length, sequence_length)
        logits = outputs[0]
        return logits

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, label, text = batch
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, label)

        # if batch_nb > 30:
        #     a, y_hat = torch.max(logits, dim=1)
        #     val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        #     val_acc = torch.tensor(val_acc)

        #     import pdb; pdb.set_trace()
        
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, label, text = batch
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, label)
        
        # acc
        a, y_hat = torch.max(logits, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}
        

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        metrics = {'val_loss': avg_val_loss, 'avg_val_acc': avg_val_acc,
                   'log': {'val_epoch_loss': avg_val_loss, 'val_epoch_acc': avg_val_acc}}
        return metrics

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, label, text, id = batch
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, label)
        
        # acc
        a, y_hat = torch.max(logits, dim=1)
        correct = label == y_hat
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())
        test_acc = torch.tensor(test_acc)

        return {'test_loss': loss, 'test_acc': test_acc,
                'id': id, 'correct': correct.tolist()}

    def test_epoch_end(self, outputs):

        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        metrics = {'test_loss': avg_test_loss, 'avg_test_acc': avg_test_acc,
                   'log': {'test_epoch_loss': avg_test_loss, 'test_epoch_acc': avg_test_acc}}

        # Save (this is coded with the ETS dataset in mind right now, so we can
        # check accuracy across different cuts
        correctness = {}
        for x in outputs:
            for i, item_id in enumerate(x['id']):
                item_correct = x['correct'][i]
                correctness[item_id] = item_correct
        save_file(correctness, self.args.eval_out_fp)

        return metrics
        
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.args.lr, eps=1e-08)

    def train_dataloader(self):
        return get_realgentext_dataloader('train', self.args.data_gen_method,
        # return get_realgentext_dataloader('train', self.args.data_gen_method,
            num_workers=4,
            batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return get_realgentext_dataloader('valid', self.args.data_gen_method,
            num_workers=4,
            batch_size=self.args.batch_size, shuffle=False)

    def test_dataloader(self):
        return get_realgentext_dataloader('test', self.args.data_gen_method,
        # return get_realgentext_dataloader('test' , self.args.data_gen_method,
            num_workers=4,
            batch_size=self.args.batch_size, shuffle=False)