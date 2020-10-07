import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class NSMCModel(nn.Module):
    def __init__(self, bert_config: BertConfig, dropout_prob: float):
        super().__init__()
        self.config = bert_config

        self.bert = BertModel(bert_config)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size, 2)

    def forward(self, input_token_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        _, pooled_outputs = self.bert.forward(input_token_ids, attention_mask, token_type_ids)
        outputs_drop = self.dropout(pooled_outputs)
        logits = self.classifier(outputs_drop)

        return logits
