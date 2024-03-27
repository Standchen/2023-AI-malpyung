import argparse
from functools import partial
import os

from datasets import Dataset
import numpy as np
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch

from utils import get_logger, init_random_seed
from preprocess import gpt_preprocess


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="nlpai-lab/kullm-polyglot-12.8b-v2", help="pre-trained model name")
    parser.add_argument("--batch-size", type=int, default=16, help="training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-4, help="initial learning rate")
    # parser.add_argument("--weight-decay", type=float, default=1e-2, help="amount of weight decay")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--rank", type=int, default=16, help="rank of LoRA matrix")
    parser.add_argument("--alpha", type=int, default=32, help="alpha in LoRA")
    parser.add_argument("--dropout", type=int, default=0.05, help="amount of dropout in LoRA")
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
    logger.info("[*] Load dataset")
    train_dataset = Dataset.from_json("data/nikluge-ea-2023-train.jsonl")

    # Labels
    labels_to_korean = {
        "joy": "기쁨",
        "anticipation": "기대",
        "trust": "신뢰",
        "surprise": "깜짝",
        "disgust": "혐오",
        "fear": "공포",
        "anger": "분노",
        "sadness": "슬픔"
    }

    # Load model and associated tokenizer.
    logger.info("[*] Load model and tokenizer")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quantization_config)

    # Though they are already the same.
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare PEFT model.
    config = LoraConfig(
        r=args.rank, 
        lora_alpha=args.alpha, 
        target_modules=["query_key_value"], 
        lora_dropout=args.dropout, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)

    # Assert all korean labels convert to a single token.
    for kor in labels_to_korean.values():
        tkns = tokenizer.tokenize(f" {kor}")
        assert len(tkns) == 1

    # Preprocess
    logger.info("[*] Prerprocessing")
    train_dataset = train_dataset.map(lambda examples: {"text": examples["input"]["form"],
                                                        "target": examples["input"]["target"]["form"] or "_",
                                                        "output": examples["output"]},
                                      remove_columns=train_dataset.column_names)
    partial_preprocess = partial(gpt_preprocess,
                                 tokenizer=tokenizer,
                                 labels_to_korean=labels_to_korean,
                                 mode="train")
    train_dataset = train_dataset.map(partial_preprocess,
                                      remove_columns=train_dataset.column_names,
                                      batched=True)

    # print(f"[!] {train_dataset[0]}")
    # for el in train_dataset[0]["input_ids"]:
    #     print(f"[!] {el} -> '{tokenizer.decode([el])}'")

    # Get colon token.
    colon_token = tokenizer(":")["input_ids"]
    assert len(colon_token) == 1
    colon_token_id = colon_token[0]
    assert isinstance(colon_token_id, int)
    logger.info(f"[*] {colon_token_id = }")

    # Custom trainer to have the training only focus on the label.
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            indices = (inputs["input_ids"] == colon_token_id).nonzero()
            for i, j in indices:
                inputs["labels"][i, :j+1] = -100
            # # Assert all rows to have desired token.
            # not_indices = np.ones(len(inputs["labels"]), dtype=int)
            # not_indices[indices[:, 0].cpu()] = 0
            # assert np.all(not_indices == 0)
            # Process.
            outputs = model(**inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            warmup_ratio=0.1,
            save_strategy="steps",
            save_steps=0.1,
            logging_steps=0.005,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            # fp16=True,
            bf16=True,
            output_dir=args.output_dir,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    # Train!
    logger.info("[*] Start training")
    trainer.train()
    logger.info("[*] Done!")
