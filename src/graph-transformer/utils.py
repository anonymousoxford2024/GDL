import argparse
import random

import numpy as np
import psutil
import torch


def set_all_seeds(seed=42):
    """Set all random seeds to a fixed value to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def memory_usage_psutil():
    process = psutil.Process()
    mem = process.memory_info().rss / float(2**20)
    return mem


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="full-attention",
        choices=["full-attention", "linformer", "performer"],
        help="Type of model to train ('full-attention', 'linformer', 'performer').",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Citeseer",
        choices=["Citeseer", "Cora", "Pubmed"],
        help="Dataset to train models on.",
    )
    args = parser.parse_args()
    return args
