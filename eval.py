import argparse
import glob
import os
from typing import Any

import torch
from tqdm.auto import tqdm

from gliner import GLiNER
from gliner.evaluation.evaluate import create_dataset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--model", type=str, default="logs/model_12000", help="Path to model folder")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to model folder")
    parser.add_argument('--data', type=str, default='data/ie_data/NER/', help='Path to the eval datasets directory')
    return parser


def get_dataset(data_paths: list[str]) -> dict[str, dict[str, Any]]:
    all_paths = glob.glob(f"{data_paths}/*")
    all_paths = sorted(all_paths)
    all_data = {}
    for p in all_paths:
        if "sample_" not in p:
            _, _, test_dataset, entity_types = create_dataset(p)
            data_name = p.split("/")[-1]

            flat_ner = True
            if any([i in data_name for i in ["ACE", "GENIA", "Corpus"]]):
                flat_ner = False

            all_data[data_name] = {"test_dataset": test_dataset, "entity_types": entity_types, "flat_ner": flat_ner}

    return all_data


@torch.no_grad()
def evaluate(test_dataset: list[dict[str, Any]], entity_types: list[str], flat_ner: bool) -> tuple[dict[str, Any], float]:
    # evaluate the model
    results, f1 = model.evaluate(test_dataset, flat_ner=flat_ner, threshold=0.5, batch_size=12, entity_types=entity_types)
    return results, f1


def get_for_all_path(model: GLiNER, steps: int, log_dir: str, data_paths: list[str]) -> None:
    all_data = get_dataset(data_paths)

    # move the model to the device
    device = next(model.parameters()).device
    model.to(device)
    # set the model to eval mode
    model.eval()

    # log the results
    save_path = os.path.join(log_dir, "results.txt")

    with open(save_path, "a") as f:
        f.write("##############################################\n")
        # write step
        f.write("step: " + str(steps) + "\n")

    zero_shot_benc = ["mit-movie", "mit-restaurant", "CrossNER_AI", "CrossNER_literature", "CrossNER_music", "CrossNER_politics", "CrossNER_science"]

    zero_shot_benc_results = {}
    all_results = {}  # without crossNER

    pbar = tqdm(total=len(list(all_data.keys())))
    for data_name, value in all_data.items():
        pbar.update(1)
        test_dataset = value['test_dataset']
        entity_types = value['entity_types']
        flat_ner = value['flat_ner']
        results, f1 = evaluate(test_dataset, entity_types, flat_ner)

        # write to file
        with open(save_path, "a") as f:
            f.write(data_name + "\n")
            f.write(str(results) + "\n")

        if data_name in zero_shot_benc:
            zero_shot_benc_results[data_name] = f1
        else:
            all_results[data_name] = f1

    avg_all = sum(all_results.values()) / len(all_results)
    avg_zs = sum(zero_shot_benc_results.values()) / len(zero_shot_benc_results)

    save_path_table = os.path.join(log_dir, "tables.txt")

    # results for all datasets except crossNER
    table_bench_all = ""
    for k, v in all_results.items():
        table_bench_all += f"{k:20}: {v:.1%}\n"
    # (20 size aswell for average i.e. :20)
    table_bench_all += f"{'Average':20}: {avg_all:.1%}"

    # results for zero-shot benchmark
    table_bench_zeroshot = ""
    for k, v in zero_shot_benc_results.items():
        table_bench_zeroshot += f"{k:20}: {v:.1%}\n"
    table_bench_zeroshot += f"{'Average':20}: {avg_zs:.1%}"

    # write to file
    with open(save_path_table, "a") as f:
        f.write("##############################################\n")
        f.write("step: " + str(steps) + "\n")
        f.write("Table for all datasets except crossNER\n")
        f.write(table_bench_all + "\n\n")
        f.write("Table for zero-shot benchmark\n")
        f.write(table_bench_zeroshot + "\n")
        f.write("##############################################\n\n")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    model = GLiNER.from_pretrained(args.model).to("cuda:0")

    get_for_all_path(model, -1, args.log_dir, args.data)
