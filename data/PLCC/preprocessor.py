from transformers import AutoTokenizer
from typing import Optional, List, Callable, Dict, Any
from tqdm import tqdm
from datasets import load_dataset, Dataset

from datapoint_base import DatapointBase
from datapoint_commit_dataset import DatapointCommitDataset


def convert_hf_to_datapoint(hf_dataset: Dataset) -> list[DatapointBase]:
    data = list()
    repos_list = list(set([hf_dp['repo'] for hf_dp in hf_dataset]))
    repos_map = {repo: repo_num for repo_num, repo in enumerate(repos_list)}

    for hf_dp in hf_dataset:
        dp = dict()
        dp['repo_name'] = hf_dp['repo']
        dp['repo_id'] = repos_map[hf_dp['repo']]
        dp['completion_lines'] = hf_dp['completion_lines']
        filenames, contents = hf_dp['repo_snapshot']['filename'], hf_dp['repo_snapshot']['content']
        assert len(filenames) == len(contents)
        dp['context_dict'] = {filename: content for filename, content in zip(filenames, contents)}
        dp['completion_dict'] = {hf_dp['completion_file']['filename']: hf_dp['completion_file']['content']}
        data.append(DatapointCommitDataset(**dp))

    return data

def prepare_data(data: List[DatapointBase], context_composer, completion_composer):
    print('Data Preparation...')
    prepared_data = list()
    for datapoint in tqdm(data):
        new_datapoint = dict()
        new_datapoint['repo_id'] = datapoint.repo_id
        new_datapoint['repo_name'] = datapoint.repo_name
        new_datapoint['completion_lines'] = datapoint.completion_lines

        new_datapoint['context'] = context_composer(datapoint)
        new_datapoint['completion'] = completion_composer(datapoint)

        prepared_data.append(type(datapoint)(**new_datapoint))

    return prepared_data

# class PreprocessorBase:
#     def __init__(self,
#                  context_composer: Callable[[Dict[str, Any]], str] | None = None,
#                  completion_composer: Callable[[Dict[str, Any]], str] | None = None,
#                  data_source: str = 'hf',
#                  ):
#         self.prepared_data: Optional[List[Dict[str, Any]]] = None
#         self.context_composer = context_composer
#         self.completion_composer = completion_composer
#         self.data_source = data_source
#
#     def compose_context(self, context: Dict[str, str]) -> str:
#         raise NotImplementedError
#
#     def compose_completion(self, context: Dict[str, str]) -> str:
#         raise NotImplementedError
#
#
#     def _datapoint_to_model_input(self, datapoint: DatapointBase) -> DatapointBase:
#         datapoint = datapoint.to_model_input(self.tokenize_datapoint)
#         return datapoint
#
#     def prepare_model_input_parallel(self, num_workers=1, dataset_path=None):
#         self.prepare_data()
#         if num_workers is None:
#             num_workers = multiprocessing.cpu_count()
#
#         if os.path.exists(dataset_path):
#             os.remove(dataset_path)
#
#         list_to_save = list()
#
#         print('Tokenization...')
#         if num_workers == 1:
#             result = [self._datapoint_to_model_input(datapoint) for datapoint in tqdm(self.prepared_data)]
#             for p in result:
#                 list_to_save.append(dataclasses.asdict(p))
#         else:
#             with Parallel(num_workers) as pool:
#                 result = pool(delayed(self._datapoint_to_model_input)(datapoint) for datapoint in self.prepared_data)
#             for p in tqdm(result):
#                 list_to_save.append(dataclasses.asdict(p))
#
#         with open(dataset_path, 'w') as json_file:
#             json.dump(list_to_save, json_file)
#
#     def save_model_inputs(self, filepath='lca/code_generation/data/model_inputs.json'):
#         with open(filepath, 'w') as f:
#             json.dump(self.prepared_data, f)
#
# class HFPreprocessor(PreprocessorBase):
#     def __init__(self, **composers):
#         super().__init__(**composers)
#         self.lang_sep_symbol = ''
#         self.meta_info_sep_symbol = 'METASEP'
#         self.extension = ''
#
#     def compose_context(self, datapoint: DatapointBase) -> str:
#         context = datapoint.context_dict
#         repo_name = datapoint.repo_name
#         # You could implement specific order of contents in composed_content
#         composed_content = [path + self.meta_info_sep_symbol + content for path, content in context.items()]
#         repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"
#         return repo_metainfo + self.lang_sep_symbol.join(composed_content)
#
#     def compose_completion(self, datapoint: DatapointBase) -> str:
#         completion = datapoint.completion_dict
#         # TODO: move path to the context
#         composed_content = [path + self.meta_info_sep_symbol + content for path, content in completion.items()]
#         return self.lang_sep_symbol + self.lang_sep_symbol.join(composed_content)
