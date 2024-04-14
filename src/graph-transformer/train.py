import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.datasets import Planetoid

from evaluation import compute_auroc
from models.full_attention import FullAttentionGraphTransformer


def train(model, data, optimizer, epochs=40, patience=5):
    train_mask = data.train_mask
    val_mask = data.val_mask

    best_val_acc = -1.0
    best_model = None
    epochs_no_improve = 0

    for epoch in range(epochs):
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
            # print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    print(f"best_val_acc = {best_val_acc}")
    return best_model if best_model is not None else model


def evaluate(model: nn.Module, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        acc = compute_auroc(logits[mask], data.y[mask])
    return loss, acc


if __name__ == "__main__":
    # # Load the dataset
    # dataset = Reddit(root='/tmp/Reddit')
    # data = dataset[0]
    # print(data)

    root = "/tmp/Citeseer"
    dataset = Planetoid(root=root, name="CiteSeer")
    data = dataset[0]

    for hidden_dim in [256]:
        for lr in [1e-3]:
            for weight_decay in [0.0]:
                for _ in range(10):
                    print(
                        f"\nhidden_dim = {hidden_dim}\t\tlr = {lr}\t\tweight_decay = {weight_decay}"
                    )
                    model = FullAttentionGraphTransformer(
                        input_dim=dataset.num_node_features,
                        hidden_dim=hidden_dim,
                        output_dim=dataset.num_classes,
                        num_layers=1,
                    )

                    optimizer = optim.Adam(
                        model.parameters(), lr=lr, weight_decay=weight_decay
                    )

                    out = model(data)
                    targets = data.y

                    trained_model = train(model, data, optimizer, epochs=120)
