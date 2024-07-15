import json
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams


class EvalVLLM:
    def __init__(
        self,
        model_name: str,
        result_folder: str | Path,
        result_filname: str = "results.jsonl",
        cache_dir: str | None = None,
        context_size: int | None = None,
    ):
        self.model = LLM(
            model=model_name,
            max_seq_len_to_capture=context_size,
            download_dir=cache_dir,
        )
        self.sampling_params = SamplingParams(
            temperature=0, max_tokens=128, stop=["\n"]
        )

        result_folder = Path(result_folder)
        result_folder.mkdir(parents=True, exist_ok=True)
        self.result_file = result_folder / result_filname
        self.result_detailed_file = self.result_file.with_stem(
            self.result_file.stem + "_detailed"
        )
        self.model_name = model_name

    def run_model(
        self, dataloader: DataLoader, limit: int = -1
    ) -> list[tuple[int, str, bool]]:
        results = []
        i = 0
        for batch in tqdm(dataloader):
            response_batch = self.model.generate(
                prompt_token_ids=batch["input_ids"],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            outputs_batch = [response.outputs[0].text for response in response_batch]

            result = self.asses_response(outputs_batch, batch["gts"])
            # TODO make it dict, not tuples, may be
            result = list(zip(batch["idx"], batch["types"], result))
            results.extend(result)
            i += 1
            if limit > 0 and i >= limit:
                break

        return results

    def eval_dataset(
        self, dataloader: DataLoader, limit: int = -1
    ) -> tuple[dict, list]:
        results = self.run_model(dataloader, limit)
        summary = self.calc_summary(results)

        return summary, results

    def eval(self, dataloaders: dict, limit: int = -1) -> list[dict]:
        for dataloader_dict in dataloaders:
            context_size = dataloader_dict["context_size"]
            context_scope = dataloader_dict["context_scope"]
            dataloader = dataloader_dict["dataloader"]
            print(
                f"Evaluating context size = {context_size} on dataset: {context_scope}"
            )
            summary_res, results = self.eval_dataset(dataloader, limit=limit)
            summary = self.save_summary(summary_res, results, dataloader_dict)
            print(summary)
        print(f"Results are saved into {self.result_file}")

        return summary

    def save_summary(
        self, summary_res: dict, results: list[list], dataloader_dict: dict
    ):
        context_size = dataloader_dict["context_size"]
        context_scope = dataloader_dict["context_scope"]
        summary = {
            "model": self.model_name,
            "context_size": context_size,
            "context_scope": context_scope,
        }
        summary.update(summary_res)
        summary_detailed = summary.copy()
        summary_detailed["details"] = results
        with open(self.result_file, "a") as f:
            json.dump(summary, f)
            f.write("\n")
        with open(self.result_detailed_file, "a") as f:
            json.dump(summary_detailed, f)
            f.write("\n")

        return summary

    @staticmethod
    def asses_response(preds: list[str], gts: list[str]) -> list[bool]:
        preds_clean = [pred.strip() for pred in preds]
        gts_clean = [gt.strip() for gt in gts]

        results = [pred == gt for pred, gt in zip(preds_clean, gts_clean)]

        return results

    @staticmethod
    def calc_summary(results: list) -> dict:
        results_total = defaultdict(lambda: defaultdict(int))

        for res in results:
            context_type = res[1]
            score = res[2]
            results_total[context_type]["correct"] += score
            results_total[context_type]["total"] += 1

        for context_type, summary in results_total.items():
            summary["ratio"] = summary["correct"] / summary["total"]
            results_total[context_type] = dict(results_total[context_type])

        return dict(results_total)
