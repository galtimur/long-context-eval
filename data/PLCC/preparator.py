import json
from pathlib import Path
from typing import List

from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

from data.PLCC.context_composer import PathDistanceComposer
from data.PLCC.datapoint_base import DatapointBase
from data.PLCC.datapoint_commit_dataset import DatapointCommitDataset


def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))
    return data


def convert_hf_to_datapoint(hf_dataset: Dataset) -> list[DatapointBase]:
    data = list()
    repos_list = list(set([hf_dp["repo"] for hf_dp in hf_dataset]))
    repos_map = {repo: repo_num for repo_num, repo in enumerate(repos_list)}

    for hf_dp in hf_dataset:
        dp = dict()
        dp["repo_name"] = hf_dp["repo"]
        dp["repo_id"] = repos_map[hf_dp["repo"]]
        dp["completion_lines"] = hf_dp["completion_lines"]
        filenames, contents = (
            hf_dp["repo_snapshot"]["filename"],
            hf_dp["repo_snapshot"]["content"],
        )
        assert len(filenames) == len(contents)
        dp["context_dict"] = {
            filename: content for filename, content in zip(filenames, contents)
        }
        dp["completion_dict"] = {
            hf_dp["completion_file"]["filename"]: hf_dp["completion_file"]["content"]
        }
        data.append(DatapointCommitDataset(**dp))

    return data


def prepare_data(data: List[DatapointBase], context_composer, completion_composer):
    print("Data Preparation...")
    prepared_data = []
    for datapoint in tqdm(data):
        new_datapoint = dict()
        new_datapoint["repo_id"] = datapoint.repo_id
        new_datapoint["repo_name"] = datapoint.repo_name
        new_datapoint["completion_path"] = list(datapoint.completion_dict.keys())[0]
        new_datapoint["context"] = context_composer(datapoint)
        new_datapoint["completion"] = completion_composer(datapoint)

        prepared_data.append(new_datapoint)

    return prepared_data


class Preparator:
    def __init__(
        self,
        data_args: DictConfig,
        dataset_name: str = "JetBrains-Research/lca-project-level-code-completion",
    ):
        self.dataset_name = dataset_name
        self.data_folder = Path(data_args.data_folder)
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.force_prepare = data_args.force_prepare

        # We do not use small context. It can cause problems.
        # Author: These are repositories that eventually became Python,
        # and we caught them at a time when Python code was not there
        self.context_sizes = [
            # "small_context",
            "medium_context",
            "large_context",
            "huge_context",
        ]
        self.composer_name = data_args.composer_name
        if self.composer_name == "path_distance":
            self.composers = PathDistanceComposer()
        else:
            raise NotImplementedError(
                f"composer {self.composer_name} is not implemented"
            )

    def get_prepared_dataset(self):
        datasets = []
        for context_size in self.context_sizes:
            dataset_filename = f"plcc_{self.composer_name}_{context_size}.jsonl"
            dataset_file = self.data_folder / dataset_filename

            if dataset_file.exists() and not self.force_prepare:
                print(f"Dataset part exists: {dataset_filename}. Loading it.")
                processed_data = read_jsonl(dataset_file)
                datasets.append(
                    {"context_size": context_size, "dataset": processed_data}
                )
                continue

            print(f"Context size - {context_size}")
            dataset_hf = load_dataset(
                self.dataset_name,
                name=context_size,
                split="test",
            )
            dataset_dp = convert_hf_to_datapoint(dataset_hf)

            processed_data = prepare_data(
                dataset_dp,
                self.composers.context_composer,
                self.composers.completion_composer,
            )
            datasets.append({"context_size": context_size, "dataset": processed_data})

            save_jsonl(processed_data, dataset_file)
            print(f"Dataset part saved to {self.data_folder}")

        return datasets
