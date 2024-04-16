from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.data.data import Data


def compute_auroc(
    logits: torch.Tensor, targets: torch.Tensor, multi_class: str = "ovr"
) -> float:
    np_probs = torch.softmax(logits, dim=1).detach().numpy()
    np_targets = targets.detach().numpy()
    if np.sum(np.isnan(np_probs)) > 0:
        return 0.0

    auroc = roc_auc_score(np_targets, np_probs, multi_class=multi_class)

    return auroc


def evaluate(model: nn.Module, data: Data, mask: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data)
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        acc = compute_auroc(logits[mask], data.y[mask])
    return loss, acc
