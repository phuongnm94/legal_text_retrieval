import argparse
from torch import nn

from transformers import BertModel, BertPreTrainedModel
 

class RelevantDocModel(BertPreTrainedModel):
    def __init__(
        self, 
        config,
        dropout
    ): 
        # init model 
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config=self.config)  # Load pretrained bert
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.classifier = nn.Linear(in_features=self.config.hidden_size, out_features=2, bias=True)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], description="Model", add_help=False)
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout value")

        # parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
        # parser.add_argument("--batch_size", type=int, default=32, help="batch size")
        # parser.add_argument("--optimizer", choices=["adamw", "sgd", "adam"], default="adam",
        #                     help="loss type")

        # parser.add_argument("--bert_config_dir", type=str, required=True, help="bert config dir")
        # parser.add_argument("--max_length", type=int, default=128, help="max length of dataset")
        # parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    
        # parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--bert_dropout", type=float, default=0.1,
        #                     help="bert dropout rate")
        # parser.add_argument("--weight_type", type=float, default=1.0)
        # parser.add_argument("--flat", action="store_true", help="is flat ner")
        # parser.add_argument("--span_loss_candidates", choices=["all", "pred_and_gold", "pred_gold_random", "gold"],
        #                     default="all", help="Candidates used to compute span loss")
        # parser.add_argument("--final_div_factor", type=float, default=1e4,
        #                     help="final div factor of linear decay scheduler")
        # parser.add_argument("--lr_mini", type=float, default=-1)
        # parser.add_argument("--ignore_index", type=int, default=-100)
        return parser

    def forward(self, input_text_pair_ids):
        outputs = self.bert(**input_text_pair_ids)
        # h_other_wordpieces = outputs[0]
        h_cls = outputs[1]  # [CLS]
        return self.classifier(self.dropout(h_cls))
 