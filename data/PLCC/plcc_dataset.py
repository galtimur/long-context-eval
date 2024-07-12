import json
import time
from functools import partial
from pathlib import Path
from typing import Dict, List
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.generation import StoppingCriteria, StoppingCriteriaList

# from eval_ppl import evaluate_ppl_red_pajamas, evaluate_base_model


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
        file_context = "\n" + item_project["completion_path"] + "\n\n" + item_completion["prefix"]
        full_context = (project_context + file_context).strip() + "\n"
        sample = {"context": full_context, "gt": item_completion["gt"], "type": item_completion["type"]}

        return sample

    @staticmethod
    def merge_context(context: Dict[str, str]) -> str:
        context_lines = []
        for key, value in context.items():
            if value == "":
                value = "# empty file"
            context_lines.extend([key, "", value, ""])  # empty string is for additional new-line

        return "\n".join(context_lines)

    @staticmethod
    def build_index(lengths: List[int]):
        index = []
        for outer_index, inner_len in enumerate(lengths):
            for inner_index in range(inner_len):
                index.append((outer_index, inner_index))
        return index


class StopOnNewLine(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = set()
        for k, tok_id in tokenizer.get_vocab().items():
            s = tokenizer.convert_tokens_to_string([k])
            if "\n" in s:
                self.stop_ids.add(tok_id)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert input_ids.shape[0] == 1  # only batch_size 1 is supported
        if input_ids[0, -1].item() in self.stop_ids:
            return True
        else:
            return False


def eval_on_lcc(
    modules: dict,
    ds_test: str | None,
    context_size: int,
    limit: int | None = None,
) -> dict:
    model, tokenizer, run_config, model_name = (
        modules["model"],
        modules["tokenizer"],
        modules["run_config"],
        modules["model_name"],
    )
    device = model.device

    stopping_criteria = StoppingCriteriaList([StopOnNewLine(tokenizer)])
    if ds_test is None:
        ds_test = LcaPythonCompletionDataset()
    max_comp = 128  # max length of the line
    # context_size = model.config.max_position_embeddings
    max_len_ctx = (
        context_size - max_comp
    )  # input context should be less that model context size minus max line length
    assert max_len_ctx > 0, "max_len_ctx should be positive!"

    grtrs = []
    preds = []

    num_samples = len(ds_test) if limit is None else limit
    ds_test = ds_test[:num_samples]
    with open(f"out/false_preds_{model_name}.txt", "a") as f:
        f.write("----- New eval -----\n")

    start_time = time.time()
    for sample in tqdm(ds_test):
        input_ids = tokenizer.encode(sample["context"], return_tensors="pt")
        input_ids = input_ids[:, -max_len_ctx:].to(device)

        model_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_comp,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": False,
        }
        if not model_name.startswith("base_model"):
            model_kwargs["segment_lengths"] = 1024

        with torch.no_grad():
            out = model.generate(**model_kwargs)

        out_tokens = out[0, len(input_ids[0]) - 1 :]
        pred = tokenizer.decode(out_tokens).strip("\n")
        preds.append(pred)
        grtrs.append(sample["gt"])
        if pred != sample["gt"]:
            with open(f"out/false_preds_{model_name}.txt", "a") as f:
                f.write(f"{pred} --> {sample['gt']}\n")

    time_used_lca = time.time() - start_time
    exact_match = sum(gt == pr for gt, pr in zip(grtrs, preds)) / len(preds)

    results = {
        "exact_match_rate": exact_match,
        "LCA items/s": num_samples / time_used_lca,
        "number of LCA items": num_samples,
    }

    return results


def eval_cross_entropy(
    modules: dict,
    dataset_ce: str | None,
    context_size: int,
    limit_loss_samples: int | None = None,
    sample_inject_function=None,
) -> dict:
    model, tokenizer, run_config, model_name = (
        modules["model"],
        modules["tokenizer"],
        modules["run_config"],
        modules["model_name"],
    )

    start_time = time.time()
    if dataset_ce is not None:
        is_auco = isinstance(model, LlamaAutoCompressorModel)
        if is_auco:
            batch_size = 8
            num_segments = (
                run_config["training_substeps"] * run_config["segments_per_substep"]
            )
            segment_length = 6 * 1024 // num_segments
        else:
            batch_size = 4
            segment_length = 1024

        eval_result = evaluate_ppl_red_pajamas(
            model,
            dataset_ce,
            batch_size,
            sample_inject_function=sample_inject_function,
            max_samples=limit_loss_samples,
            split_size=segment_length,
            context_size=context_size,
            is_auco=is_auco,
            disable_tqdm=False,
        )
        # , context_size=context_size

        num_chunks = eval_result["num_chunks"]
        av_loss = eval_result["total_loss"]
        first_chunk_loss = eval_result[f"chunk_0_loss"]
        last_chunk_loss = eval_result[f"chunk_{num_chunks-1}_loss"]

    time_used_loss = time.time() - start_time
    results = {
        "loss": av_loss,
        "last chunk loss": last_chunk_loss,
        "first chunk loss": first_chunk_loss,
        "loss items/s": limit_loss_samples / time_used_loss,
        "number of loss items": limit_loss_samples,
    }

    return results


def run_benchmark(
    ckpt_map_path: str | Path,
    results_path: str | Path,
    limit: int | None = None,
    limit_loss_samples: int | None = None,
    do_lcc: bool = True,
    do_ce_loss: bool = True,
    copy_last_chunck: bool = False,
):
    with open(ckpt_map_path, "r") as f:
        ckpt_name_map = json.load(f)
    dataset_lcc = LcaPythonCompletionDataset()
    dataset_ce = load_from_disk(
        "/mnt/data2/shared-data/autocompressors/6k_py_320000_samp/valid"
    )
    dataset_ce = dataset_ce.shuffle(seed=42)
    for model_name, ckpt_path in ckpt_name_map.items():
        eval_kwargs = {
            "checkpoint_path": ckpt_path,
            "ds_test": dataset_lcc,
            "dataset_ce": dataset_ce,
            "model_name": model_name,
            "limit": limit,
            "limit_loss_samples": limit_loss_samples,
            "sample_inject_function": dummy_inject,
            "do_lcc": do_lcc,
            "do_ce_loss": do_ce_loss,
        }

        if copy_last_chunck:
            for n in range(5):
                model_name_list = model_name.split("_")
                model_name_list.insert(2, f"n_{n}_")
                model_name_current = "_".join(model_name_list)
                copy_segment_n = partial(copy_segment, segment_size=1024, n_to_copy=n)

                eval_kwargs["model_name"] = model_name_current
                eval_kwargs["sample_inject_function"] = copy_segment_n

                print(f"Running {model_name_current}")
                eval_result = eval_model(**eval_kwargs)
                with open(results_path, "a") as jsonl_file:
                    jsonl_file.write(json.dumps(eval_result) + "\n")
        else:
            print(f"Running {model_name}")
            eval_result = eval_model(**eval_kwargs)
            with open(results_path, "a") as jsonl_file:
                jsonl_file.write(json.dumps(eval_result) + "\n")


if __name__ == "__main__":
    ckpt_map_path = "configs/ckpt_name_map.json"
    results_path = "out/eval_lca_cc.json"
