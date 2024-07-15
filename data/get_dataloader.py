from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.data_collator import DataCollator
from data.PLCC.get_datasets import get_datasets
from data.PLCC.plcc_dataset import PLCCDataset


class DataloadersFetcher:
    def __init__(self, config: DictConfig):
        self.data_args = config.data
        self.eval_args = config.eval
        self.context_sizes = self.eval_args.context_sizes
        # TODO move context tokenization to the datasets to reduce redundant work
        #  But be careful with sos and other special tokens

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    def get_dataloader(self, dataset: PLCCDataset, context_size: int) -> DataLoader:
        data_collator = DataCollator(
            context_size=context_size,
            batch_size=self.eval_args.batch_size,
            tokenizer=self.tokenizer,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.eval_args.batch_size,
            collate_fn=data_collator.collate_fn,
            shuffle=False,
        )

        return dataloader

    def get_dataloaders(self) -> list[dict]:
        datasets = get_datasets(self.data_args)
        dataloaders = []
        for context_size in self.context_sizes:
            for context_scope, dataset in datasets.items():
                dataloader_dict = {
                    "context_size": context_size,
                    "context_scope": context_scope,
                    "dataloader": self.get_dataloader(dataset, context_size),
                }
                dataloaders.append(dataloader_dict)
        return dataloaders
