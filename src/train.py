import argparse
import glob
from pytorch_lightning import Trainer

import torch
from torch.utils.data.dataloader import DataLoader
from model import RelevantDocClassifier
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from pytorch_lightning.callbacks import ModelCheckpoint


class ColieePreprocessor:
    def __init__(self, tokenizer, max_seq_length) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, mini_batch):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length)

        question_ids = [e[1] for e in mini_batch]
        c_ids = [e[2] for e in mini_batch]
        questions = [e[3] for e in mini_batch]
        c_codes = [e[4] for e in mini_batch]
        input_text_pair_ids = self.tokenizer(questions, c_codes, padding='max_length', 
                                    max_length=max_seq_length, truncation=True, return_tensors='pt')

        labels = torch.LongTensor([e[0] for e in mini_batch])

        return (input_text_pair_ids, labels, question_ids, c_ids)

if __name__=="__main__":

    # training+model args
    parser = argparse.ArgumentParser(description="Training Args")
    parser = RelevantDocClassifier.add_model_specific_args(parser)
        
    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--log_dir", type=str, default=".", help="log dir")
    parser.add_argument("--max_keep_ckpt", default=1, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--pretrained_checkpoint", default=None, type=str, help="pretrained checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--model_name_or_path", type=str, help="pretrained model name or path")
    parser.add_argument("--ignore_index",  type=int, default=-100)
    parser.add_argument("--max_epochs", default=5, type=int, help="Max training epochs.")
    parser.add_argument("--max_seq_length",  type=int, default=512, help="Max seq length for truncating.")
    parser.add_argument("--no_train", action="store_true", default=False, help="Do not training.")
    parser.add_argument("--no_test", action="store_true", default=False, help="Do not test.")
    parser.add_argument("--gpus", nargs='+', default=[0], type=int, help="Id of gpus for training")
    parser.add_argument("--ckpt_steps", default=1000, type=int, help="number of training steps for each checkpoint.")

    opts = parser.parse_args()
    if opts.pretrained_checkpoint is not None and not opts.pretrained_checkpoint.endswith(".ckpt"):
        opts.pretrained_checkpoint = glob.glob(f"{opts.pretrained_checkpoint}/*.ckpt")[0]
        print(f"Found checkpoint - {opts.pretrained_checkpoint}")

    # load pretrained_checkpoint if it is set 
    if opts.pretrained_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(opts.log_dir, use_fast=True, config=AutoConfig.from_pretrained(opts.log_dir))
        model = RelevantDocClassifier.load_from_checkpoint(opts.pretrained_checkpoint, tokenizer=tokenizer)
        max_seq_length=model.args.max_seq_length
    else:
        config = AutoConfig.from_pretrained(opts.model_name_or_path)
        config.save_pretrained(opts.log_dir)
        tokenizer = AutoTokenizer.from_pretrained(opts.model_name_or_path, use_fast=True, max_seq_length=opts.max_seq_length)
        tokenizer.save_pretrained(opts.log_dir)
        max_seq_length=opts.max_seq_length

    #
    # Data loader 
    coliee_data_preprocessor = ColieePreprocessor(tokenizer, max_seq_length=max_seq_length)
    df_train = pd.read_csv(f"{opts.data_dir}/train.csv")
    train_loader = DataLoader(df_train.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)
    df_dev = pd.read_csv(f"{opts.data_dir}/dev.csv")
    dev_loader = DataLoader(df_dev.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)

    # model 
    if not opts.pretrained_checkpoint: 
        model = RelevantDocClassifier(opts, tokenizer=tokenizer, data_train_size=len(train_loader))
    
    # trainer
    checkpoint_callback = ModelCheckpoint(dirpath=opts.log_dir, save_top_k=opts.max_keep_ckpt, 
                                          auto_insert_metric_name=True, mode="max", monitor="valid_f2", 
                                        #   every_n_train_steps=opts.ckpt_steps
                                          )
    trainer = Trainer(max_epochs=opts.max_epochs, 
                      accelerator='gpu' if len(opts.gpus) > 0 else 'cpu', 
                      devices=opts.gpus, 
                      callbacks=[checkpoint_callback], 
                      default_root_dir=opts.log_dir, 
                      val_check_interval=0.1
                      )

    if not opts.no_train:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    if not opts.no_test:
        df_test = pd.read_csv(f"{opts.data_dir}/test.csv")
        test_loader = DataLoader(df_test.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)
        trainer.test(model, test_loader)

    # df_test = pd.read_csv(f"{opts.data_dir}/test.csv")
    # test_loader = DataLoader(df_test.values, batch_size=opts.batch_size, collate_fn=coliee_data_preprocessor, shuffle=True)
    # x = trainer.predict(model, test_loader,return_predictions=True, ckpt_path=opts.pretrained_checkpoint)
    # print(x)