from typing import Dict, List

import torch
from transformers import AutoTokenizer


class DataCollator:
    def __init__(
        self,
        context_len: int,
        batch_size: int,
        tokenizer: AutoTokenizer,
        context_key: str = "context",
    ) -> None:
        self.batch_size = batch_size
        self.buffer = []
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        self.context_len = context_len
        self.context_key = context_key

    def collate_fn(self, batch: List[Dict]) -> torch.Tensor:
        batch_context = [item[self.context_key] for item in batch]
        tokenized_batch = self.tokenizer(
            batch_context,
            truncation=True,
            padding="max_length",
            max_length=self.context_len,
            return_tensors="pt",
        )

        return tokenized_batch["input_ids"]
