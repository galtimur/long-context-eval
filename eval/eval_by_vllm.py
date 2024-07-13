from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams


class EvlaVLLM:
    def __init__(self, model_name: str, context_len: int):
        self.model = LLM(
            model=model_name,
            max_seq_len_to_capture=context_len,
            download_dir="/mnt/data2/tmp",
        )
        self.sampling_params = SamplingParams(
            temperature=0, max_tokens=128, stop=["\n"]
        )

    def run_model(self, dataloader: DataLoader) -> list[tuple[int, str, bool]]:
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
            if i > 10:
                break

        return results

    def eval(self, dataloader):
        results = self.run_model(dataloader)
        summary = self.calc_summary(results)

        return summary, results

    @staticmethod
    def asses_response(preds: list[str], gts: list[str]) -> list[bool]:
        preds_clean = [pred.strip() for pred in preds]
        gts_clean = [gt.strip() for gt in gts]

        results = [pred == gt for pred, gt in zip(preds_clean, gts_clean)]

        return results

    @staticmethod
    def calc_summary(results):
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
