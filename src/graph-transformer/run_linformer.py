import argparse

import numpy as np
from torch import optim
from torch_geometric.datasets import Planetoid

from models.linformer_attention import LinformerGraphTransformer
from train import train, evaluate, parse_arguments
from utils import set_all_seeds

if __name__ == "__main__":
    set_all_seeds(0)

    parser = argparse.ArgumentParser()
    args = parse_arguments()
    dataset_name = args.dataset

    root = f"/tmp/{dataset_name}"
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]
    n_epochs = 200

    print(f"dataset = {dataset}")

    for hidden_dim in [512]:
        for feature_dim in [64, 128, 256]:
            for lr in [1e-3]:
                for weight_decay in [0.0]:
                    print(
                        f"\nhidden_dim = {hidden_dim}\tfeature_dim={feature_dim}\tlr = {lr}\tweight_decay = {weight_decay}"
                    )

                    auroc_scores = []
                    time_measurements = []
                    memory_measurements = []

                    for i in range(10):

                        input_dim = dataset.num_node_features
                        output_dim = dataset.num_classes
                        num_layers = 1

                        model = LinformerGraphTransformer(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            num_layers=num_layers,
                            projection_dim=feature_dim,
                            n_nodes=data.num_nodes,
                        )

                        optimizer = optim.Adam(
                            model.parameters(), lr=lr, weight_decay=weight_decay
                        )

                        results = train(
                            model,
                            data,
                            optimizer,
                            n_epochs=n_epochs,
                            patience=5,
                            verbose=False,
                        )
                        model = results["model"]

                        val_loss, val_acc = evaluate(model, data, data.val_mask)
                        # print(f"val_acc = {val_acc}")

                        test_loss, test_acc = evaluate(model, data, data.test_mask)
                        # print(f"test_acc = {test_acc}")

                        auroc_scores.append(test_acc)
                        time_measurements.append(results["time_mean"])
                        memory_measurements.append(results["memory_mean"])

                    auroc_mean = np.mean(auroc_scores)
                    auroc_std = np.std(auroc_scores)
                    print(
                        f"Mean AUROC: {100 * auroc_mean:.2f} +- {100 * auroc_std:.2f}"
                    )
                    time_mean = np.mean(time_measurements)
                    memory_mean = np.mean(memory_measurements)
                    print(f"time_mean = {time_mean}")
                    print(f"memory_mean = {memory_mean}")
