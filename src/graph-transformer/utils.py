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
