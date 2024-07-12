from omegaconf import OmegaConf
from vllm import LLM, SamplingParams

from data.get_dataloader import DataloadersFetcher

config_path = "configs/config.yaml"
if config_path is None:
    config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
data_args, model_args, eval_args = config.data, config.model, config.eval
# config = OmegaConf.to_container(config, resolve=True)

# TODO add context lengths to config
# TODO add check that the real context length is larger that config value

dl_fetcher = DataloadersFetcher(config)
dataloaders = dl_fetcher.get_dataloaders()

dataloader = dataloaders["medium_context"]
dl_iter = iter(dataloader)
item = next(dl_iter)
texts = dl_fetcher.tokenizer.batch_decode(item, skip_special_tokens=True)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=128,
    stop=["\n"]
)

model = LLM(model=model_args.model_name)

# response = model.generate(prompts="def hello_world():")
response1 = model.generate(prompt_token_ids=item[0].tolist()[-2000:], sampling_params=sampling_params)
output1 = response1[0].outputs[0].text

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=128,
)
response2 = model.generate(prompt_token_ids=item[0].tolist()[-2000:], sampling_params=sampling_params)
output2 = response2[0].outputs[0].text

print(output1)
print(output2)

pass
