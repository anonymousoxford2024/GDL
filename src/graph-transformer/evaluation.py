import torch
from sklearn.metrics import roc_auc_score


def compute_auroc(
    logits: torch.Tensor, targets: torch.Tensor, multi_class: str = "ovr"
) -> float:
    np_probs = torch.softmax(logits, dim=1).detach().numpy()
    np_targets = targets.detach().numpy()

    auroc = roc_auc_score(np_targets, np_probs, multi_class=multi_class)

    return auroc
