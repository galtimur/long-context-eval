"""
File taken from and edited
https://github.com/JetBrains-Research/lca-baselines/blob/main/project_level_code_completion/composers/
"""

import os
from typing import Dict, List

from data.PLCC.datapoint_base import DatapointBase


class PathDistanceComposer:
    def __init__(
        self,
        lang_extension=".py",
        allowed_extensions=[".md"],
        filter_extensions=True,
        completion_types=["infile", "inproject"],
    ):
        self.lang_extension = lang_extension
        self.allowed_extensions = allowed_extensions + [lang_extension]
        self.filter_extensions = filter_extensions
        self.completion_types = completion_types

    def filter_paths(self, list_of_filepaths):
        filtered_lists = [
            file
            for file in list_of_filepaths
            if any(file.endswith(ext) for ext in self.allowed_extensions)
        ]
        return filtered_lists

    @staticmethod
    def _path_distance(path_from, path_to):
        divided_path_from = os.path.normpath(path_from).split(os.path.sep)
        divided_path_to = os.path.normpath(path_to).split(os.path.sep)
        common_len = 0
        for el1, el2 in zip(divided_path_from, divided_path_to):
            if el1 == el2:
                common_len += 1
            else:
                break
        return (len(divided_path_from) - common_len - 1) + (
            len(divided_path_to) - common_len - 1
        )

    def _sort_filepaths(self, path_from, list_of_filepaths):
        max_len = max(
            [
                len(os.path.normpath(path).split(os.path.sep))
                for path in list_of_filepaths
            ]
        )
        max_len += len(os.path.normpath(path_from).split(os.path.sep))
        paths_by_distance = [list() for _ in range(max_len)]

        for path_to in list_of_filepaths:
            dist = self._path_distance(path_from, path_to)
            paths_by_distance[dist].append(path_to)
        return [path for path_group in paths_by_distance for path in path_group]

    def context_composer(self, datapoint: DatapointBase) -> str:
        context = datapoint.get_context()
        completion = datapoint.get_completion()
        assert len(completion) == 1, "Only one file should be completed"
        completion_path = list(completion)[0]
        sorted_paths = self._sort_filepaths(completion_path, list(context))
        if self.filter_extensions:
            sorted_paths = self.filter_paths(sorted_paths)
        # ! Important ! Most relevant files are at the end!
        sorted_paths = sorted_paths[::-1]
        context = {path: context[path] for path in sorted_paths}

        return context

    def completion_composer(
        self, datapoint: DatapointBase
    ) -> Dict[str, List[List[str]]]:
        """
        returns dict completion_type: completion_list
        completion_list: list of tuples (ground_truth_line, completion_prefix)
        """

        completion = datapoint.completion_dict
        assert len(completion) == 1, "Only one file should be completed"
        content = list(completion.values())[0].split("\n")
        completions = dict()
        for completion_type in self.completion_types:
            lines = datapoint.completion_lines[completion_type]
            gt = [content[line] for line in lines]
            prefixes = ["\n".join(content[:line]) for line in lines]
            completions[completion_type] = [list(pair) for pair in zip(gt, prefixes)]

        return completions
