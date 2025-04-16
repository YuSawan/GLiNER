import ast
import json
import random
import re
import string
import sys
from collections import defaultdict
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm

ANONYMIZE = sys.argv[1] # label, mention, or both


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


def anonymize(tokens: list[str]) -> list[str]:
    randlist = []
    for token in tokens:
        randomized_chars = [random.choice(string.ascii_letters + string.digits) for i in range(len(token))]
        randlist.append(''.join(randomized_chars))
    return randlist


def process_data(data: Dataset, anonym_type: str) -> list[dict[str, list[Any]]]:
    """Processes a list of data entries to extract entity spans."""
    all_data = []
    mention2anom, anom2mention = defaultdict(str), defaultdict(str)
    label2anom, anom2label = defaultdict(str), defaultdict(str)
    for entry in tqdm(data, total=len(data)):
        p_data = extract_entity_spans(entry)
        tokens = p_data["tokenized_text"]
        ner = []
        for s, e, label in p_data["ner"]:
            mention = tokens[s: e+1]
            if anonym_type in ['mention', 'both']:
                if " ".join(mention) not in mention2anom.keys():
                    while True:
                        anonymized_mention = ' '.join(anonymize(mention))
                        if len(anonymized_mention) == 1:
                            anom2mention[anonymized_mention] = " ".join(mention)
                            mention2anom[" ".join(mention)] = anonymized_mention
                            break
                        if anonymized_mention not in anom2mention.keys():
                            anom2mention[anonymized_mention] = " ".join(mention)
                            mention2anom[" ".join(mention)] = anonymized_mention
                            break
                tokens[s: e+1] = anonymized_mention.split(' ')
            if anonym_type in ["label", "both"]:
                if label not in label2anom.keys():
                    while True:
                        anonymized_label = ' '.join(anonymize(label.split(' ')))
                        if anonymized_label not in anom2label.keys():
                            anom2label[anonymized_label] = label
                            label2anom[label] = anonymized_label
                            break
                label = anonymized_label
            ner.append((s, e, label))
        all_data.append({"tokenized_text": tokens, "ner": ner, "negative": []})
    return all_data


def save_data_to_file(data:list[dict[str, list[Any]]], filepath: str) -> None:
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    data = load_dataset('Universal-NER/Pile-NER-type')
    processed_data = process_data(data['train'], anonym_type = ANONYMIZE)
    print(processed_data[0])
    print("dataset size:", len(processed_data))
    save_data_to_file(processed_data, f'pilener_train_anonymize_{ANONYMIZE}.jsonl')

