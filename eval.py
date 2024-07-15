from omegaconf import OmegaConf

from data.get_dataloader import DataloadersFetcher
from eval.eval_by_vllm import EvlaVLLM

config_path = "configs/config.yaml"
if config_path is None:
    config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
# config = OmegaConf.to_container(config, resolve=True)

# TODO add context lengths to config
# TODO add check that the real context length is larger that config value
# TODO Make a context length independent of the tokenizer.

dl_fetcher = DataloadersFetcher(config)
dataloaders = dl_fetcher.get_dataloaders()

dataloader = dataloaders[0]
# dl_iter = iter(dataloader)
# item = next(dl_iter)
# texts = dl_fetcher.tokenizer.batch_decode(item, skip_special_tokens=True)

# TODO correct, now context size is a list of contexts
evaluator = EvlaVLLM(config.model.model_name, config.eval.context_size)
summary, results = evaluator.eval(dataloader)

pass
