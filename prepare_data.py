from omegaconf import OmegaConf

from data.PLCC.get_datasets import get_datasets
from data.data_collator import DataCollator

config_path = "configs/config.yaml"
if config_path is None:
    config_path = "configs/config.yaml"
config = OmegaConf.load(config_path)
data_args = config.data
model_args = config.model
eval_args = config.eval
# config = OmegaConf.to_container(config, resolve=True)

datasets = get_datasets(data_args)

dataset = datasets["medium_context"]
item = dataset[2]

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
data_collator = DataCollator(
    context_len=eval_args.context_len,
    batch_size=eval_args.batch_size,
    tokenizer=tokenizer,
)

dataloader = DataLoader(
    dataset,
    batch_size=eval_args.batch_size,
    collate_fn=data_collator.collate_fn,
)

dl_iter = iter(dataloader)
item = next(dl_iter)

texts = tokenizer.batch_decode(item, skip_special_tokens=True)

pass
