import argparse
from functools import partial
import os

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import get_logger, init_random_seed, compute_metrics
from preprocess import bert_preprocess


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="klue/roberta-large", help="pre-trained model name")
    parser.add_argument("--train-batch-size", type=int, default=24, help="training batch size")
    parser.add_argument("--val-batch-size", type=int, default=48, help="validation batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="amount of weight decay")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="output directory")
    args = parser.parse_args()

    # Create output directory.
    os.makedirs(args.output_dir)

    # Get logger.
    logger = get_logger(filepath=os.path.join(args.output_dir, "output.log"))

    # List parsed arguments.
    logger.info("########## Arguments ##########")
    for k, v in vars(args).items():
        logger.info(f"{k:20}: {v}")

    # Initialize random seed.
    init_random_seed(seed=args.seed)

    # Load datasets.
    logger.info("[*] Load datasets")
    train_dataset = Dataset.from_json("data/nikluge-ea-2023-train.jsonl")
    val_dataset = Dataset.from_json("data/nikluge-ea-2023-dev.jsonl")

    # Labels
    labels = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}

    # Load pre-trained model and associated tokenizer.
    logger.info("[*] Load pre-trained model and associated tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logger.info("[*] Preprocessing")
    partial_preprocess = partial(bert_preprocess, tokenizer=tokenizer, labels=labels)
    train_dataset = train_dataset.map(partial_preprocess, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(partial_preprocess, remove_columns=val_dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model= "f1",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train!
    logger.info("[*] Start training")
    trainer.train()
    logger.info("[*] Done!")
