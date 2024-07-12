"""
File taken from
https://github.com/JetBrains-Research/lca-baselines/blob/main/project_level_code_completion/data_classes/datapoint_base.py
"""

import dataclasses


@dataclasses.dataclass
class DatapointBase:
    repo_id: int
    repo_name: str
    completion_lines: dict[str, list[int]]
    context_dict: dict[str, str]
    # context_dict: keys are filepaths, values are file contents
    completion_dict: dict[str, str]
    context: str
    completion: str
    context_len: int
    model_input: list[int]
