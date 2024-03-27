import argparse
from collections import OrderedDict
from functools import partial
import json
import os

from datasets import Dataset
import numpy as np
from peft import PeftConfig, PeftModel
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_logger
from preprocess import gpt_preprocess


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt-path", type=str, required=True, help="path to the model check point")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--output-dir", type=str, default="./inference", help="output directory")
    args = parser.parse_args()

    # Create output directory.
    os.makedirs(args.output_dir)

    # Get logger.
    logger = get_logger(filepath=os.path.join(args.output_dir, "output.log"))

    # List parsed arguments.
    logger.info("########## Arguments ##########")
    for k, v in vars(args).items():
        logger.info(f"{k:20}: {v}")

    # Load datasets.
    logger.info("[*] Load datasets")
    val_dataset = Dataset.from_json("data/nikluge-ea-2023-dev.jsonl")

    # Labels
    labels_to_korean = OrderedDict({
        "joy": "기쁨",
        "anticipation": "기대",
        "trust": "신뢰",
        "surprise": "깜짝",
        "disgust": "혐오",
        "fear": "공포",
        "anger": "분노",
        "sadness": "슬픔"
    })

    # Load model and associated tokenizer.
    logger.info("[*] Load model and tokenizer")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    peft_config = PeftConfig.from_pretrained(args.model_ckpt_path)

    logger.info(f"[*] Base model: {peft_config.base_model_name_or_path = }")
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, quantization_config=quantization_config)
    model = PeftModel.from_pretrained(model, args.model_ckpt_path)

    # Though they are already the same.
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocessing
    logger.info("[*] Prerprocessing")
    val_dataset = val_dataset.map(lambda examples: {"text": examples["input"]["form"],
                                                    "target": examples["input"]["target"]["form"] or "_",
                                                    "output": examples["output"]},
                                  remove_columns=val_dataset.column_names)

    partial_preprocess = partial(gpt_preprocess,
                                 tokenizer=tokenizer,
                                 labels_to_korean=labels_to_korean,
                                 mode="val")
    val_dataset = val_dataset.map(partial_preprocess,
                                  remove_columns=val_dataset.column_names,
                                  batched=True)

    # Sort according to the length of `input_ids` for fast inference.
    # val_dataset = val_dataset.map(lambda examples: {"length" : len(examples["input_ids"])})  \
    #                          .sort("length")  \
    #                          .remove_columns("length")
    val_dataset = val_dataset.with_format("torch")

    # Inference settings.
    logger.info("[*] Move the model to CUDA memory and switch to eval mode")
    model.to(torch.device("cuda"))
    model.eval()
    model.config.use_cache = True
    logger.info("[*] Disable gradient")
    torch.set_grad_enabled(False)

    # Inference
    logger.info("[*] Now inference")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))

    labels_to_ids = OrderedDict(
        {label: tokenizer(f" {kor}")["input_ids"][0] for label, kor in labels_to_korean.items()}
    )

    logits = []
    for batch in tqdm(val_dataloader):
        # Fetch the batch and move it to CUDA.
        batch = {k: v.to("cuda") for k, v in batch.items()}
        # Get the outputs of model.
        out = model.generate(
            **batch,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # Compute transition scores.
        batch_logits = np.zeros((len(out.sequences), len(labels_to_ids)))
        for i, id_ in enumerate(labels_to_ids.values()):
            out.sequences[:, -1] = id_
            transition_scores = model.compute_transition_scores(out.sequences,
                                                                out.scores,
                                                                normalize_logits=True)
            batch_logits[:, i] = transition_scores[:, -1].cpu().numpy()
        logits.append(batch_logits)

    # Aggregate batched logits.
    logits = np.vstack(logits)

    # Calculate probabilities.
    probs = np.exp(logits)
    probs /= probs.sum(axis=1)[:, None]

    # Predict.
    pred_labels = (probs > 0.5).tolist()

    # Read true labels.
    with open("data/nikluge-ea-2023-dev.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.read().strip().split("\n")]
    true_labels = []
    for line in data:
        true_labels.append([1 if line["output"][label] == "True" else 0 for label in labels_to_korean])

    # Metrics
    f1 = f1_score(y_true=true_labels, y_pred=pred_labels, average="micro")
    roc_auc = roc_auc_score(y_true=true_labels, y_score=pred_labels, average="micro")
    accuracy = accuracy_score(y_true=true_labels, y_pred=pred_labels)

    logger.info(f"[*] {f1 = :.3f}")
    logger.info(f"[*] {roc_auc = :.3f}")
    logger.info(f"[*] {accuracy = :.3f}")
