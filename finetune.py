import argparse
import glob
import os
import random
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch

from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.evaluation.evaluate import create_dataset
from gliner.training import Trainer, TrainingArguments


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--model", type=str, default="logs/model_12000", help="Path to model folder")
    parser.add_argument("--output_dir", type=str, default="logs", help="Path to model folder")
    parser.add_argument('--data', type=str, default='data/ie_data/NER/', help='Path to the eval datasets directory')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=5000)
    return parser


def get_dataset(data_paths: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_paths = glob.glob(f"{data_paths}/*")
    all_paths = sorted(all_paths)
    all_data = []
    for p in all_paths:
        if "sample_" not in p:
            train_dataset, _, _, _ = create_dataset(p)
            random.shuffle(train_dataset)
            all_data.extend(train_dataset[:10000])

    print('Dataset size:', len(all_data))
    print('Dataset is shuffled...')
    random.shuffle(all_data)
    train_dataset = all_data[:int(len(all_data)*0.9)]
    test_dataset = all_data[int(len(all_data)*0.9):]

    return train_dataset, test_dataset


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    train_dataset, test_dataset = get_dataset(args.data)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = GLiNER.from_pretrained(args.model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear", #cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        save_steps = 100,
        save_total_limit=10,
        dataloader_num_workers = 0,
        use_cpu = False if torch.cuda.is_available() else True,
        report_to="none",
    )

    # use it for better performance, it mimics original implementation but it's less memory efficient
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
