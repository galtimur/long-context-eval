from omegaconf import OmegaConf

from data.PLCC.preparator import Preparator
from data.PLCC.plcc_dataset import PLCCDataset

config_path = "configs/config.yaml"
if config_path is None:
    config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
data_args = config.data
# config = OmegaConf.to_container(config, resolve=True)

preparator = Preparator(data_args)
prepared_datasets = preparator.get_prepared_dataset()
dataset = PLCCDataset(prepared_datasets[0]["dataset"])

item = dataset[2]

pass
