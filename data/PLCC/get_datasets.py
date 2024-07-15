from omegaconf import DictConfig

from data.PLCC.plcc_dataset import PLCCDataset
from data.PLCC.preparator import Preparator


def get_datasets(data_args: DictConfig) -> dict[str, PLCCDataset]:
    preparator = Preparator(data_args)
    prepared_datasets = preparator.get_prepared_datasets()
    datasets = dict()

    for prepared_dataset in prepared_datasets:
        context_scope = prepared_dataset["context_scope"]
        datasets[context_scope] = PLCCDataset(prepared_dataset["dataset"])

    return datasets
