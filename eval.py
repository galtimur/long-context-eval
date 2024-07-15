from omegaconf import OmegaConf

from data.get_dataloader import DataloadersFetcher
from eval.eval_by_vllm import EvalVLLM

config_path = "configs/config.yaml"
if config_path is None:
    config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
# config = OmegaConf.to_container(config, resolve=True)

# TODO add context lengths to config
# TODO add check that the real context length is larger that config value
# TODO Make a context length independent of the tokenizer.
# TODO add speed to the output metrics
# TODO cut the context before tokenization or even gathering (cut on the fly). Otherwise it takes too much time
# Write a cross-entropy metric for the model

dl_fetcher = DataloadersFetcher(config)
dataloaders = dl_fetcher.get_dataloaders()

# dataloader = dataloaders[0]
# dl_iter = iter(dataloader)
# item = next(dl_iter)
# texts = dl_fetcher.tokenizer.batch_decode(item, skip_special_tokens=True)

evaluator = EvalVLLM(
    config.model.model_name,
    result_folder=config.output.result_folder,
    cache_dir=config.output.cache_dir,
)
summary = evaluator.eval(dataloaders, limit=3)
