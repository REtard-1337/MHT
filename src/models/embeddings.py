import torch
import torch.nn as nn
from transformers import BertModel
from typing import Tuple

class EmbeddingLayer(nn.Module):
    def __init__(self) -> None:
        super(EmbeddingLayer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
