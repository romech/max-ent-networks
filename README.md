# Multiplex Network Reconstruction using Entropy Maximisation Algorithms

Code accompanying master's thesis.

Author: *Roman Aleksandrov*

Title: *"Development of a Link Reconstruction Method for Multiplex Networks"*, (University of Amsterdam, 2021).

A pdf link will be added later.

Note that there is another Python package [NEMtropy](https://github.com/nicoloval/NEMtropy) which is probably a better solution for out-of-the-box single-layer network reconstruction.

## Data

Datasets used:

1. *FAO Multiplex*. [[Homepage]](https://manliodedomenico.com/data.php) [[Paper]](https://www.nature.com/articles/ncomms7864) [[Data]](https://manliodedomenico.com/data/FAO_Multiplex_Trade.zip). Unzip the data into `FAO_dataset` folder to run experiments.
2. *Dutch Business Network*. Private access.

## Algorithms

The following algorithms are implemented:

1. MaxEnt
2. Iterative Proportional Fitting (IPF) + modifications
3. Fitness-induced Directed Binary Configuration Model (f-DMCM) + modifications
4. **Multiplexity fitting**

## Installation

Tested with Python 3.6 and 3.8, package list attached in requirements.txt.

## Evaluation of single-layer reconstruction

Code of the experiments is available for the FAO Network only. The scripts save resulting plots and tables into `output` folder (you might need to create it in advance).

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

## Evaluation of multi-layer reconstruction (multiplexity fitting)

Similarly, the experiments with two-layer reconstruction using IPF + multiplexity fitting can be done repeated with:
```
python -m experiments.fao_multireconstruct --all
```

Other possible options are: `--single`, `-n N`.

In addition, visualisation script is included. Need to set target file and desired plots in the script code.
```
python -m experiments.visualise_multiplexity
```

## Description of contents

Starting point is the **reconstruction** folder. Algorithm implementations are placed there.

**experiments** folder contains:

- **fao_reconstruct.py** -- single-layer reconstruction for the FAO network
- **fao_multireconstruct.py** -- multi-layer experiments
- **visualise_*.py** -- various visualisations appearing in the paper

**fao_data.py**, **sampling.py** -- data loading functions for the FAO network.

**multiplex** folder contains helper functions for multi-layer experiments.

**fao_analysis** contains scripts for descriptive analysis of the network.
