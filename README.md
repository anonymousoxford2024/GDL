# Geometric Deep Learning

## Evaluating the Impact of Linearized Attention vs Full-Attention in Graph Transformers

This project aims to investigate the impact of linearized attention mechanisms compared
to the conventional full-attention mechanism.
Specifically we explore Linformer and Performer adaptations from NLP to the graph domain.

## Getting started:

Start by cloning the repository:

```
git clone https://github.com/anonymousoxford2024/GDL.git
```

Or via SSH:

```
git clone git@github.com:anonymousoxford2024/GDL.git
```

If you don't already have Poetry installed, by running the following command.
More information on this can be found in the
[Poetry Documentation](https://python-poetry.org/docs/).

```
pip install poetry
poetry install --all-extras
```

Activate the virtual environment with:

```
poetry shell
```

You're now ready to replicate the study's findings.

## Reproduce results:

Running the following commands will retrain the specified model on a dataset of 
your choice.

Set the model using `--model-type {MODELTYPE}` to be one of 
`["full-attention", "linformer", "performer"]`. 
It defaults to `full-attention"`.

Set the dataset using `--dataset {DATASET}` to be one of 
`["Citeseer", "Cora", "Pubmed"]`.
It defaults to `Citeseer"`.



For example, the following retrains a linformer model on Citeseer:
```
python src/graph-transformer/run_train.py  --model-type linformer --dataset Citeseer
```


The evaluation scores are scored in `plots/plotting_{DATASET}.json`. 
To run the evaluation and plot the results, run:

```
python src/graph-transformer/plotting.py  --dataset {DATASET}
```

