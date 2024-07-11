import os
from datasets import load_dataset

from context_composer import PathDistanceComposer
from preprocessor import convert_hf_to_datapoint, prepare_data

# def preprocess():
#%%
dataset_hf = load_dataset(
    "JetBrains-Research/lca-project-level-code-completion",
    name="medium_context",
    split="test",
)
dataset_dp = convert_hf_to_datapoint(dataset_hf)
#%%

composers = PathDistanceComposer()
processed_data = prepare_data(dataset_dp, composers.context_composer, composers.completion_composer)
pass
#%%

# prepared_dataset_path = os.path.join(, f'model_inputs_composer_path_dist_comp.json')

# preprocessor = HFPreprocessor(
#     dataset_params=args.dataset,
#     context_composer=PathDistanceComposer
# )
# preprocessor.prepare_model_input_parallel(dataset_path=prepared_dataset_path, num_workers=1)  # Don't change num_workers
