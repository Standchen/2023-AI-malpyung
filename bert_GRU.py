import argparse
from functools import partial
import os

from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_logger, init_random_seed, compute_metrics
from preprocess import bert_preprocess


class CustomClassifier(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int,
                 hidden_size: int = 512, num_layers: int = 2):
        super().__init__()
        # GRU
        self.gru = nn.GRU(input_size=emb_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=0.3,
                          bidirectional=True)
        # Fully-connected layers
        feature_dim = (2 * num_layers * hidden_size) + emb_dim
        hidden_dim = 2 * feature_dim
        self.fc_1 = nn.Linear(feature_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, out_dim)
        # Dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, model_outputs):
        last_h = model_outputs.last_hidden_state
        batch_size = last_h.shape[0]
        # GRU
        _, h_n = self.gru(last_h)
        # Concatenate GRU outputs to CLS token embedding.
        features = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        features = torch.cat([features, last_h[:, 0, :]], dim=-1)

        features = F.relu(self.fc_1(features))
        features = self.dropout(features)

        features = F.relu(self.fc_2(features))
        features = self.dropout(features)

        features = self.fc_3(features)
        return features


class CustomModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 8):
        super().__init__()
        # Base model (pre-trained)
        self.base_model = AutoModel.from_pretrained(model_name)
        # Custom layers (need to be trained)
        emb_dim = self.base_model.embeddings.word_embeddings.embedding_dim
        self.classifier = CustomClassifier(emb_dim=emb_dim, out_dim=num_labels)

    def forward(self, input_ids, labels, attention_mask, token_type_ids):
        # Get outputs from base model.
        out = self.base_model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              output_hidden_states=True)
        # Get logits from classifier.
        logits = self.classifier(out)
        # Compute loss.
        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )



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

    # Load pre-trained model and associated tokenizer.
    logger.info("[*] Load customized pre-trained model and associated tokenizer")
    model = CustomModel(args.model, num_labels=len(labels)).to("cuda")
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
