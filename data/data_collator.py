from transformers import AutoTokenizer


class DataCollator:
    def __init__(
        self,
        context_size: int,
        batch_size: int,
        tokenizer: AutoTokenizer,
        context_key: str = "context",
        ground_truth_key: str = "gt",
        metainfo_key: str = "type",
    ) -> None:
        self.batch_size = batch_size
        self.buffer = []
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        self.context_size = context_size
        self.context_key = context_key
        self.ground_truth_key = ground_truth_key
        self.metainfo_key = metainfo_key

    def collate_fn(self, batch: list[dict]) -> dict[str, list]:
        batch_ids = [item["id"] for item in batch]
        batch_context = [item[self.context_key] for item in batch]
        batch_gt = [item[self.ground_truth_key] for item in batch]
        batch_types = [item[self.metainfo_key] for item in batch]

        tokenized_batch = self.tokenizer(
            batch_context,
            truncation=True,
            padding="max_length",
            max_length=self.context_size,
            # return_tensors="pt",
        )

        return {
            "input_ids": tokenized_batch["input_ids"],
            "gts": batch_gt,
            "idx": batch_ids,
            "types": batch_types,
        }
