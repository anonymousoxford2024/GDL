import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data.data import Data
from torch_geometric.datasets import Planetoid

from evaluation import compute_auroc
from models.full_attention import FullAttentionGraphTransformer
from models.linformer_attention import LinformerGraphTransformer
from models.performer_attention import PerformerGraphTransformer
from utils import set_all_seeds


def train(
    model: nn.Module,
    data: Data,
    optimizer: optim.Optimizer,
    n_epochs: int = 40,
    patience: int = 5,
):
    train_mask = data.train_mask
    val_mask = data.val_mask

    best_val_acc = -1.0
    best_model = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Compute the AUROC for train and validation sets
        train_acc = compute_auroc(out[train_mask], data.y[train_mask])
        val_loss, val_acc = evaluate(model, data, val_mask)

        print(
            f"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    print(f"best_val_acc = {best_val_acc}")
    return best_model if best_model is not None else model


def evaluate(model: nn.Module, data: Data, mask: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data)
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        acc = compute_auroc(logits[mask], data.y[mask])
    return loss, acc


def parse_arguments() -> argparse.Namespace:
    parser.add_argument(
        "--model_type",
        type=str,
        default="performer",
        choices=["full-attention", "linformer", "performer"],
        help="Type of model to train ('full-attention' or 'linformer').",
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


if __name__ == "__main__":
    set_all_seeds(0)
    # # Load the dataset
    # dataset = Reddit(root='/tmp/Reddit')
    # data = dataset[0]
    # print(data)

    parser = argparse.ArgumentParser()

    args = parse_arguments()
    model_type = args.model_type
    dataset_name = args.dataset

    root = f"/tmp/{dataset_name}"
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]
    n_epochs = 200

    print(f"model_type = {model_type}")
    print(f"dataset = {dataset}")
    print(f"data.num_nodes = {data.num_nodes}")

    for hidden_dim in [256]:
        for lr in [1e-3]:
            for weight_decay in [0.0]:
                for feature_dim in [64]:
                    # for _ in range(10):
                    print(
                        f"\nhidden_dim = {hidden_dim}\t\tlr = {lr}\t\tweight_decay = {weight_decay}"
                    )

                    input_dim = dataset.num_node_features
                    output_dim = dataset.num_classes
                    num_layers = 1

                    if model_type == "full-attention":
                        model = FullAttentionGraphTransformer(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=num_layers,
                        )
                    elif model_type == "linformer":
                        # Example configuration and model initialization
                        projection_dim = feature_dim

                        model = LinformerGraphTransformer(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=num_layers,
                            projection_dim=projection_dim,
                            n_nodes=data.num_nodes,
                        )
                    elif model_type == "performer":
                        # features_dim = 256
                        model = PerformerGraphTransformer(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=num_layers,
                            num_features=feature_dim,
                        )

                    optimizer = optim.Adam(
                        model.parameters(), lr=lr, weight_decay=weight_decay
                    )

                    trained_model = train(
                        model, data, optimizer, n_epochs=n_epochs, patience=10
                    )
