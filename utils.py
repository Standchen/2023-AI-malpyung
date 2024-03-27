import logging
import random
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
import torch.nn.functional as F


def get_logger(filepath: str):
    """Get logger."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    # Formatter
    formatter = logging.Formatter("[%(asctime)s] (%(module)s) %(message)s")
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    # File handler
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # Add handlers.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def init_random_seed(seed: int, strict: bool = False):
    # Python random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Metrics
def compute_metrics(eval_preds: EvalPrediction):
    out = F.sigmoid(torch.Tensor(eval_preds.predictions))
    y_pred = (out > 0.5).numpy().astype(int)
    f1 = f1_score(y_true=eval_preds.label_ids, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true=eval_preds.label_ids, y_score=y_pred, average="micro")
    accuracy = accuracy_score(y_true=eval_preds.label_ids, y_pred=y_pred)
    return {"f1": f1,
            "roc_auc": roc_auc,
            "accuracy": accuracy}
