from typing import Dict
from omegaconf import DictConfig

from data.PLCC.preparator import Preparator
from data.PLCC.plcc_dataset import PLCCDataset

def get_datasets(data_args: DictConfig) -> Dict[str, PLCCDataset]:

    preparator = Preparator(data_args)
    prepared_datasets = preparator.get_prepared_dataset()
    datasets = dict()

    for prepared_dataset in prepared_datasets:
        context_size = prepared_dataset["context_size"]
        datasets[context_size] = PLCCDataset(prepared_dataset["dataset"])

    return datasets
