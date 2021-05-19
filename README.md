# Network Reconstruction with Entropy Maximisation Algorithms

Just source code yet, without description.

## Data

Datasets used:

1. *FAO Multiplex*. [[Homepage]](https://manliodedomenico.com/data.php) [[Paper]](https://www.nature.com/articles/ncomms7864) [[Data]](https://manliodedomenico.com/data/FAO_Multiplex_Trade.zip). Unzip the data into `FAO_dataset` folder to run experiments.
2. *Dutch Business Network*. Private access.

## Algorithms

The following algorithms are implemented:

1. MaxEnt
2. Iterative Proportional Fitting (IPF) + modifications
3. Fitness-induced Directed Binary Configuration Model (f-DMCM) + modifications

## Installation

Tested with Python 3.6 and 3.8, package list attached in requirements.txt.

## Running evaluation

Code of experiments is available for FAO network only. The scripts save resulting plots and tables into `output` folder (you might need to create it in advance).

Running for some scpecific layer by id to assess quality and generate reconstruction heatmap:
```
python -m experiments.fao_reconstruct -l 31
```

Running for `n` random layers with `s` different random seeds to get score tables quickly:
```
python -m experiments.fao_reconstruct -n 10 -s 3
```

Running for all layers:
```
python -m experiments.fao_reconstruct -a
```

Once evaluation for all layers is done, creating plots for F1-scores and constraints is possible with:
```
python -m experiments.visualise_scores
```
