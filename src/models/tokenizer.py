from transformers import BertTokenizer
from typing import List, Dict, Any


class Tokenizer:
    def __init__(self, model_name: str = "bert-base-multilingual-cased") -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(self, texts: List[str]) -> Dict[str, Any]:
        tokens = self.tokenizer.batch_encode_plus(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        return tokens
