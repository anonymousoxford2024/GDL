import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import parse_arguments

PROJECT_DIR = Path(__file__).parent.parent.parent


def extract_model_data(json_data):
    models = []
    dimensions = []
    aurocs = []
    errors = []
    times = []
    memory = []

    for model_name, auroc_data in json_data["auroc"].items():
        if isinstance(auroc_data, dict):
            for dimension, auroc_value in auroc_data.items():
                models.append(model_name)
                dimensions.append(dimension)
                aurocs.append(auroc_value)
                errors.append(json_data["error"][model_name][dimension])
                times.append(json_data["time_mean"][model_name][dimension])
                memory.append(json_data["memory_mean"][model_name][dimension])
        else:
            models.append(model_name)
            dimensions.append("-")
            aurocs.append(auroc_data)
            errors.append(json_data["error"][model_name])
            times.append(json_data["time_mean"][model_name])
            memory.append(json_data["memory_mean"][model_name])

    return models, dimensions, aurocs, errors, times, memory


def plot_auroc_performances(models, dims, aurocs, errors, dataset):
    unique_colors = {
        "Full-Attention": "#2ca02c",
        "Linformer": "#1f77b4",
        "Performer": "#ff7f0e",
    }
    colors = [unique_colors[model_name] for model_name in models]

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.35
    index = np.arange(len(models))
    bars = []
    for i, model in enumerate(models):
        bar = ax.bar(
            index[i],
            aurocs[i],
            yerr=errors[i],
            color=colors[i],
            capsize=5,
            width=bar_width,
            label=model if model not in [b.get_label() for b in bars] else "",
        )
        bars.append(bar)

    ax.set_xlabel("Models and Feature Dimensions", fontsize=12)
    ax.set_ylabel("AUROC in %", fontsize=12)
    ax.set_title(f"Model Performances on {dataset} Dataset", fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(
        [f"{model}\n{dim}" for model, dim in zip(models, dims)], fontsize=12
    )

    if dataset == "Citeseer":
        ax.set_ylim([75, 85])
    elif dataset == "Cora":
        ax.set_ylim([80, 90])
    elif dataset == "Pubmed":
        ax.set_ylim([80, 90])

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")

    handles, labels = ax.get_legend_handles_labels()
    unique_handles = [
        handles[labels.index(l)] for l in sorted(set(labels), key=labels.index)
    ]
    ax.legend(
        unique_handles, [l for l in sorted(set(labels), key=labels.index)], fontsize=12
    )

    plt.tight_layout()

    plt.savefig(PROJECT_DIR / "plots" / f"performance_comparison_{dataset}.png")
    plt.show()


def plot_memory_and_time_consumptions(models, dims, times, memory, dataset):
    unique_colors = {
        "Full-Attention": "#660000",
        "Linformer": "#ff0000",
        "Performer": "#ff9999",
    }
    colors = [unique_colors[model_name] for model_name in models]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bar_width = 0.35
    index = np.arange(len(models))
    bars = []
    for i, model in enumerate(models):
        bar = ax1.bar(
            index[i],
            memory[i],
            color=colors[i],
            capsize=5,
            width=bar_width,
            label=model if model not in [b.get_label() for b in bars] else "",
        )
        bars.append(bar)

    ax1.set_xlabel("Models and Feature Dimensions", fontsize=12)
    ax1.set_ylabel("Memory Usage (MB)", color="red", fontsize=12)
    ax1.set_title(f"Time and Memory Usage on {dataset} Dataset", fontsize=15)
    ax1.set_xticks(index)
    ax1.set_xticklabels(
        [f"{model}\n{dim}" for model, dim in zip(models, dims)], fontsize=12
    )

    if dataset == "Citeseer":
        ax1.set_ylim([50, 110])
    elif dataset == "Cora":
        ax1.set_ylim([10, 40])
    elif dataset == "Pubmed":
        ax1.set_ylim([40, 140])

    # secondary y-axis for time data
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time (seconds)", color="blue", fontsize=12)
    ax2.plot(index, times, "x", color="blue")
    ax2.tick_params(axis="y", colors="blue")
    ax2.set_ylim([0, 20])
    if dataset == "Pubmed":
        ax2.set_ylim([0, 300])

    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    unique_handles = handles1 + handles2
    unique_labels = labels1 + labels2
    ax1.legend(unique_handles, unique_labels, loc="upper left", fontsize=12)

    plt.tight_layout()

    plt.savefig(PROJECT_DIR / "plots" / f"memory_usage_comparison_{dataset}.png")
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = args.dataset

    with open(str(PROJECT_DIR / f"plots/plotting_{dataset.lower()}.json"), "r") as file:
        data = json.load(file)

    models, dims, aurocs, errors, times, memory = extract_model_data(data)

    plot_auroc_performances(models, dims, aurocs, errors, dataset)
    plot_memory_and_time_consumptions(models, dims, times, memory, dataset)
