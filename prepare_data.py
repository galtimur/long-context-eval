from omegaconf import OmegaConf

from data.get_dataloader import DataloadersFetcher

config_path = "configs/config.yaml"
if config_path is None:
    config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
data_args, model_args, eval_args = config.data, config.model, config.eval
# config = OmegaConf.to_container(config, resolve=True)

dl_fetcher = DataloadersFetcher(config)
dataloaders = dl_fetcher.get_dataloaders()
dataloader = dataloaders["medium_context"]

dl_iter = iter(dataloader)
item = next(dl_iter)

texts = dl_fetcher.tokenizer.batch_decode(item, skip_special_tokens=True)

pass
