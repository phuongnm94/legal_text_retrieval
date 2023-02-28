
import os
import argparse
import logging
from torch import nn

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import SGD
from model.relevant_doc_model import RelevantDocModel
from data_utils.utils import set_random_seed
from transformers import BertConfig, BertModel, BertPreTrainedModel
import torch.nn.functional as F
 
set_random_seed(0)

class RelevantDocClassifier(pl.LightningModule):
    def __init__(
        self, 
        args: argparse.Namespace,
        tokenizer,
        data_train_size=None
    ):
        """Initialize."""
        super().__init__() 

        format = '%(asctime)s - %(name)s - %(message)s'
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
            logging.basicConfig(format=format, filename=os.path.join(self.args.log_dir, "run.log"), level=logging.INFO)
        else:
            args = argparse.Namespace(**args)
            self.args = args
            logging.basicConfig(format=format, filename=os.path.join(self.args.log_dir, "run.log"), level=logging.INFO)

        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
    
        self.ignore_index = args.ignore_index

        # init model 
        # Load config from pretrained name or path 
        self.config = BertConfig.from_pretrained(args.model_name_or_path)  # Load pretrained bert
        self.model = RelevantDocModel.from_pretrained(args.model_name_or_path, dropout=self.args.dropout)
        self.tokenizer = tokenizer

        self.optimizer = self.args.optimizer
        self.data_train_size = data_train_size


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], description="Model", add_help=False)
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout value")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--lr_scheduler", type=str, default="onecycle", ) 
        parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
        parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps used for scheduler.")
        parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="accumulate_grad_batches.")

        parser.add_argument("--optimizer", choices=["adamw", "sgd", "adam"], default="adam",
                            help="loss type")
        parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay if we apply some.")

        # parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")

        # parser.add_argument("--bert_config_dir", type=str, required=True, help="bert config dir")
        # parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    
        # parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--bert_dropout", type=float, default=0.1,
        #                     help="bert dropout rate")
        # parser.add_argument("--weight_type", type=float, default=1.0)
        # parser.add_argument("--flat", action="store_true", help="is flat ner")
        # parser.add_argument("--span_loss_candidates", choices=["all", "pred_and_gold", "pred_gold_random", "gold"],
        #                     default="all", help="Candidates used to compute span loss")
        # parser.add_argument("--lr_mini", type=float, default=-1)
        # parser.add_argument("--ignore_index", type=int, default=-100)
        return parser


    def training_step(self, batch, batch_idx, return_y_hat=False):
        input_text_pair_ids, labels, question_ids, c_ids  = batch
        y_hat = self.model(input_text_pair_ids)
        loss = F.cross_entropy(y_hat, labels, ignore_index=self.args.ignore_index)
        if return_y_hat:
            return loss, y_hat
        return loss

    def validation_step(self, batch, batch_idx):
        input_text_pair_ids, labels, question_ids, c_ids  = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        self.log("val_batch_loss",loss, prog_bar=True)
        return {'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels, 'question_ids': question_ids, 'c_ids': c_ids}
    
    @staticmethod
    def group_by_qid(question_ids, c_ids, relevants):
        results = {}
        for i, question_id in enumerate(question_ids):
            if question_id not in results:
                results[question_id] = {'c_ids': [], 'question_id': question_id, 'f2': 0.0}
            if bool(relevants[i]):
                results[question_id]['c_ids'].append(c_ids[i])
        return results

    def validation_epoch_end(self, batch_parts, no_log_tensorboard=False, main_prediction_enss=None):
        miss_q, main_prediction = main_prediction_enss if main_prediction_enss is not None else (None, None)
        
        # aggregrate values 
        def aggregrate_val(batch_parts):
            predictions = torch.cat([torch.argmax(batch_output['y_hat'], dim=1) for batch_output in batch_parts],  dim=0)
            labels = torch.cat([batch_output['labels']  for batch_output in batch_parts],  dim=0)
            question_ids, c_ids = [], []
            for batch_output in batch_parts:
                question_ids += batch_output['question_ids']
                c_ids += batch_output['c_ids']

            gr_pred = self.group_by_qid(question_ids, c_ids, predictions)
            gr_gold = self.group_by_qid(question_ids, c_ids, labels)
            return gr_pred, gr_gold
        
        gr_pred, gr_gold =  aggregrate_val(batch_parts)
        if main_prediction is not None:
            main_gr_pred, main_gr_gold =  aggregrate_val(main_prediction)
            for k in list(gr_pred.keys()):
                if k not in miss_q:
                    gr_pred[k] = main_gr_pred[k]

        def f2(p, r):
            if 4*p + r == 0:
                return 0
            return (5*p*r)/(4*p + r)

        for q_id, gold_info in gr_gold.items():
            gold_c_ids = list(set(gold_info['c_ids']))
            pred_c_ids = list(set(gr_pred[q_id]['c_ids']))
            count_true = 0 
            for c_id in pred_c_ids:
                if c_id in gold_c_ids:
                    count_true += 1
            gold_info['pred_c_ids'] = pred_c_ids
            gold_info['retrieved'] = count_true 
            gold_info['p'] = count_true / len(pred_c_ids) if len(pred_c_ids) > 0 else 0.0
            gold_info['r'] = count_true / len(gold_c_ids)if len(gold_c_ids) > 0 else 0.0
            gold_info['f2'] = f2(gold_info['p'], gold_info['r'])

        retrieved = sum([e['retrieved'] for k, e in gr_gold.items()])
        return_results = {'retrieved': retrieved} 

        for metric in ['p', 'r', 'f2']:
            _values = [e[metric] for k, e in gr_gold.items()]
            avg = sum(_values) / len(_values)
            return_results[f'valid_{metric}'] = avg
            
        if not no_log_tensorboard:
            self.log("retrieved", retrieved, prog_bar=True)
            self.log("valid_f2", return_results['valid_f2'], prog_bar=True)
        self.result_logger.info(f"total_q = {len(gr_gold)}" )

        # collect miss query
        missed_q = [k for k, v in gr_pred.items() if len(v['c_ids']) == 0]
        return_results['miss_q'] = missed_q
        self.result_logger.info(f"Miss_querry = {missed_q}" )
        if not no_log_tensorboard:
            self.log("count_missed_q", len(missed_q))

        return return_results

    def predict_step(self, batch, batch_idx):
        input_text_pair_ids, labels, question_ids, c_ids  = batch
        y_hat = self.model(input_text_pair_ids)
        return {'y_hat': y_hat, 'question_ids': question_ids, 'c_ids': c_ids, 'labels': labels}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, batch_parts):
        result = self.validation_epoch_end(batch_parts)
        self.result_logger.info(f"Retrieved = {result['retrieved']}, f2 = {result['valid_f2']}")
        return result

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        elif self.optimizer == "adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len(self.args.gpus)
        t_total = int((self.data_train_size // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        elif self.args.lr_scheduler == "polydecay":
            if self.args.lr_mini == -1:
                lr_mini = self.args.lr / 5
            else:
                lr_mini = self.args.lr_mini
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, self.args.warmup_steps, t_total, lr_end=lr_mini)
        else:
            raise ValueError
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
