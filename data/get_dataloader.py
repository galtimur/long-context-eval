from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.data_collator import DataCollator
from data.PLCC.get_datasets import get_datasets


class DataloadersFetcher:
    def __init__(self, config):
        self.data_args = config.data
        self.eval_args = config.eval

        # TODO move context tokenization to the datasets to reduce redundant work
        #  But be carefull with sos and other special tokens

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        self.data_collator = DataCollator(
            context_len=self.eval_args.context_len,
            batch_size=self.eval_args.batch_size,
            tokenizer=self.tokenizer,
        )

    def get_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.eval_args.batch_size,
            collate_fn=self.data_collator.collate_fn,
        )

        return dataloader

    def get_dataloaders(self):
        datasets = get_datasets(self.data_args)
        dataloaders = dict()
        for context_size, dataset in datasets.items():
            dataloaders[context_size] = self.get_dataloader(dataset)

        return dataloaders
