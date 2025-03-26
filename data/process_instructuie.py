import glob
import json
import random
import sys
from typing import Any

from gliner.evaluation.evaluate import create_dataset

random.seed(1000)


def process_data(data_paths: list[str]) -> list[dict[str, Any]]:
    all_paths = glob.glob(f"{data_paths}/*")
    all_paths = sorted(all_paths)
    all_data = []
    data_cnt = 1
    for p in all_paths:
        if "sample_" not in p:
            data_name = p.split("/")[-1]
            if data_name in ["CrossNER_AI", "CrossNER_literature", "CrossNER_music", "CrossNER_politics", "CrossNER_science", "ACE 2004"]:
                continue
            train_dataset, _, _, _ = create_dataset(p)
            random.shuffle(train_dataset)
            all_data.extend(train_dataset[:10000])
            print(f"({data_cnt}) {data_name}: {len(train_dataset[:10000])}")
            data_cnt += 1
    return all_data


def save_data_to_file(data:list[dict[str, list[Any]]], filepath: str) -> None:
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    processed_data = process_data(sys.argv[1])
    save_data_to_file(processed_data, 'instructuie_train.json')
    print("dataset size:", len(processed_data))
