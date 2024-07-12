from omegaconf import OmegaConf

from data.PLCC.get_datasets import get_datasets

config_path = "configs/config.yaml"
if config_path is None:
    config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
data_args = config.data
# config = OmegaConf.to_container(config, resolve=True)

datasets = get_datasets(data_args)

item = datasets["large_context"][2]

pass
