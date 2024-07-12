from typing import Dict, List

from torch.utils.data import Dataset


class PLCCDataset(Dataset):
    def __init__(self, prepared_dataset) -> None:
        self.prepared_dataset = prepared_dataset
        self.samples = []
        self.lengths = [len(sample["completion"]) for sample in prepared_dataset]
        self.len = sum(self.lengths)
        self.index = self.build_index(self.lengths)
        pass

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> dict[str, str]:
        return self.get_sample_by_index(idx)

    def get_sample_by_index(self, idx):
        idx_in, idx_out = self.index[idx]
        item_project = self.prepared_dataset[idx_in]
        project_context = self.merge_context(item_project["context"])
        item_completion = item_project["completion"][idx_out]
        file_context = (
            "\n" + item_project["completion_path"] + "\n\n" + item_completion["prefix"]
        )
        full_context = (project_context + file_context).strip() + "\n"
        sample = {
            "context": full_context,
            "gt": item_completion["gt"],
            "type": item_completion["type"],
        }

        return sample

    @staticmethod
    def merge_context(context: Dict[str, str]) -> str:
        context_lines = []
        for key, value in context.items():
            if value == "":
                value = "# empty file"
            context_lines.extend(
                [key, "", value, ""]
            )  # empty string is for additional new-line

        return "\n".join(context_lines)

    @staticmethod
    def build_index(lengths: List[int]):
        index = []
        for outer_index, inner_len in enumerate(lengths):
            for inner_index in range(inner_len):
                index.append((outer_index, inner_index))
        return index
