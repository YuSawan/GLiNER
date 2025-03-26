import ast
import json
import re
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm


def tokenize_text(text: str) -> list[str]:
    """Tokenizes the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def extract_entity_spans(entry: dict[str, Any]) -> dict[str, list[Any]]:
    """Extracts entity spans from an entry."""
    len_start = len("What describes ")
    len_end = len(" in the text?")
    entity_types, entity_texts, negative = [], [], []

    for c in entry['conversations']:
        if c['from'] == 'human' and c['value'].startswith('Text: '):
            text = c['value'][len('Text: '):]
            tokenized_text = tokenize_text(text)
        elif c['from'] == 'human' and c['value'].startswith('What describes '):
            entity_type = c['value'][len_start:-len_end]
            entity_types.append(entity_type)
        elif c['from'] == 'gpt' and c['value'].startswith('['):
            if c['value'] == '[]':
                negative.append(entity_types.pop())
                continue
            texts_ents = ast.literal_eval(c['value'])
            entity_texts.extend(texts_ents)
            num_repeat = len(texts_ents) - 1
            entity_types.extend([entity_types[-1]] * num_repeat)

    entity_spans = []
    for j, entity_text in enumerate(entity_texts):
        entity_tokens = tokenize_text(entity_text)
        matches = []
        for i in range(len(tokenized_text) - len(entity_tokens) + 1):
            if " ".join(tokenized_text[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                matches.append((i, i + len(entity_tokens) - 1, entity_types[j]))
        if matches:
            entity_spans.extend(matches)

    return {"tokenized_text": tokenized_text, "ner": entity_spans, "negative": negative}


def process_data(data: Dataset) -> list[dict[str, list[Any]]]:
    """Processes a list of data entries to extract entity spans."""
    all_data = [extract_entity_spans(entry) for entry in tqdm(data)]
    return all_data


def save_data_to_file(data:list[dict[str, list[Any]]], filepath: str) -> None:
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    data = load_dataset('Universal-NER/Pile-NER-type')
    processed_data = process_data(data['train'])
    save_data_to_file(processed_data, 'pilener_train.json')

    print("dataset size:", len(processed_data))
