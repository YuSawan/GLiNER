import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any


def load_data(filepath: os.PathLike) -> list[dict[str, Any]]:
    """Loads data from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def tokenize_text(text: str) -> list[str]:
    """Tokenizes the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def process_entities(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_data = []
    for data in dataset:
        tokenized_text = tokenize_text(data['sentence'])
        entity_spans = []
        for entity in data['entities']:
            start = entity['pos'][0]
            if start == 0:
                entity_spans.append((0, len(tokenize_text(entity['name'])) - 1, entity['type']))
            else:
                prefix = tokenize_text(data['sentence'][:start])
                flag = False
                for i in range(1, len(tokenized_text)):
                    if tokenized_text[:i] == prefix:
                        flag = True
                        entity_spans.append((i, i + len(tokenize_text(entity['name'])) - 1, entity['type']))
                        break
                if not flag:
                    warnings.warn(f"removed entities due to substring ``{entity['name']}''([{entity['pos'][0]}, {entity['pos'][1]}]) in ``{data['sentence']}''")
        all_data.append({"tokenized_text": tokenized_text, "ner": entity_spans})

    return all_data


def process_data(dataset_dir: str, output_dir: str) -> None:
    dirs = Path(dataset_dir).glob("*")
    for dir in dirs:
        output_dirpath = Path(output_dir) / dir.name
        output_dirpath.mkdir(parents=True, exist_ok=True)

        filepaths = [dir / fname for fname in ['train.json', 'dev.json', 'test.json']]
        assert filepaths[0].exists() and filepaths[1].exists() and filepaths[2].exists()
        for filepath in filepaths:
            print(dir.name, filepath.name)
            output_fpath = output_dirpath / filepath.name
            data = load_data(filepath)
            processed_data = process_entities(data)
            save_data_to_file(processed_data, output_fpath)


def save_data_to_file(data: list[dict[str, Any]], filepath: os.PathLike) -> None:
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", "-d", required=True, metavar="DIR", help="IE_INSTRUCTIONS/NER/")
    parser.add_argument("--output_dir", "-o", required=True, metavar="DIR")

    args = parser.parse_args()
    process_data(args.dataset_dir, args.output_dir)
