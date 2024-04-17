import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data.data import Data
from torch_geometric.datasets import Planetoid

from evaluation import compute_auroc, evaluate
from models.full_attention import FullAttentionGraphTransformer
from models.linformer_attention import LinformerGraphTransformer
from models.performer_attention import PerformerGraphTransformer
from utils import set_all_seeds, memory_usage_psutil, parse_arguments


def train(
    model: nn.Module,
    data: Data,
    optimizer: optim.Optimizer,
    n_epochs: int = 40,
    patience: int = 5,
    verbose: bool = True,
):
    train_mask = data.train_mask
    val_mask = data.val_mask

    best_val_acc = -1.0
    epochs_no_improve = 0
    best_model_state = None

    time_measurements = []
    memory_measurements = []

    for epoch in range(n_epochs):
        start_time = time.time()
        start_mem = memory_usage_psutil()

        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()

        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        train_acc = compute_auroc(out[train_mask], data.y[train_mask])
        val_loss, val_acc = evaluate(model, data, val_mask)

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_measurements.append(elapsed_time)

        end_mem = memory_usage_psutil()
        mem_usage = end_mem - start_mem
        memory_measurements.append(mem_usage)

        if verbose:
            print(
                f"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        # check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    time_mean = np.mean(time_measurements)
    memory_mean = np.mean(memory_measurements)

    model.load_state_dict(best_model_state)
    return {"model": model, "time_mean": time_mean, "memory_mean": memory_mean}


if __name__ == "__main__":
    set_all_seeds(0)

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

    feat_dims = [64, 128, 256]
    if model_type == "full-attention":
        feat_dims = [None]
        
    for hidden_dim in [512]:
        for lr in [1e-3]:
            for weight_decay in [0.0]:
                for feature_dim in feat_dims:
                    for _ in range(10):
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
