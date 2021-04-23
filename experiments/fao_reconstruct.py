import logging
import random
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toolz
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from experiments.metrics import binary_classification_metrics
from fao_data import load_dataset
from reconstruction import dbcm, ipf, random_baseline
from sampling import LayerSplit, layer_split_with_no_observables, random_layer_split
from utils import (edges_to_matrix, filter_by_layer, highlight_first_line,
                   index_elements, matrix_intersetions, probabilies_to_adjacency)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def fao_layer_sample(layer_id=None, hidden_ratio=0.5, random_state=None) -> LayerSplit:
    """
    Select random layer, then split its nodes into 'observed' and 'hidden' parts.
    Edges that are adjacent to any hidden node are also considered as 'hidden'.

    Args:
        layer_id (int, optional): Chosen at random by default.
        hidden_ratio (float): Ratio of nodes hidden (0.25 by default).
                              Ratio of hidden edges is essentially larger.
                              If ratio = 1.0 then the observed edge list is empty, and
                              reconstruction algorithm does not account for them.
        random_state (int): Random seed for splitting nodes.

    Returns:
        LayerSplit: data class containing nodes and edges for each partition:
                    'observed', 'hidden' and 'full'.
    """
    dataset = load_dataset()
    if layer_id is None:
        layer_id = random.choice(dataset.layer_names.index)
    
    edges = filter_by_layer(dataset.edges, layer_id)
    
    if hidden_ratio != 1:
        return random_layer_split(
            edges=edges,
            layer_id=layer_id,
            hidden_ratio=hidden_ratio,
            random_state=random_state
        )
    else:
        return layer_split_with_no_observables(
            edges=edges,
            layer_id=layer_id
        )


def evaluate_reconstruction(
        sample: LayerSplit,
        probability_matrix: np.ndarray,
        verbose: bool = False):
    n = len(sample.node_index)
    assert probability_matrix.shape == (n, n)
    
    # Target (hidden) edges. Converting to zero-based index.
    target_edges_src = toolz.get(sample.hidden.edges.node_1.tolist(), sample.node_index)
    target_edges_dst = toolz.get(sample.hidden.edges.node_2.tolist(), sample.node_index)
    target_edges = list(zip(target_edges_src, target_edges_dst))
    
    # Assigning zero probabilities to edges between observed nodes
    observed_entries = matrix_intersetions(sample.observed.nodes, index=sample.node_index)
    probability_matrix = probability_matrix.copy()
    probability_matrix[observed_entries] = 0
    np.fill_diagonal(probability_matrix, 0)
    
    # Transforming probabilities into adjacency matrix and edge list
    pred_matrix = probabilies_to_adjacency(probability_matrix)
    pred_edges_src, pred_edges_dst = pred_matrix.nonzero()
    pred_edges = list(zip(pred_edges_src, pred_edges_dst))
    
    metrics = binary_classification_metrics(target_edges, pred_edges)
    if verbose:
        print(pd.Series(metrics))
    return metrics


def demo_evaluate_all_layers():
    layer_ids = load_dataset(drop_small_layers=True).layer_names.index
    # layer_ids = np.arange(1, 30)
    seeds = [10, 42]
    experiments = [
        ('Random', random_baseline.reconstruct_layer_sample),
        ('IPF', ipf.reconstruct_layer_sample),
        ('DBCM', dbcm.reconstruct_layer_sample),
    ]
    
    index_keys = []
    runs = []
    for layer_id in layer_ids:
        for seed in seeds:
            sample = fao_layer_sample(layer_id, random_state=seed)
            for name, reconstruct_func in experiments:
                index_keys.append((layer_id, name, seed))
                runs.append((sample, reconstruct_func))
    
    results_list = process_map(_run_single_eval, runs, chunksize=3, max_workers=6)
    results_df = pd.DataFrame(
        results_list,
        index=pd.MultiIndex.from_tuples(index_keys, names=['layer_id', 'name', 'seed']),
    )
    metrics = ['f1', 'precision', 'recall']
    stats_by_layer = results_df[metrics].groupby(level=['layer_id', 'name']).agg(_describe)
    stats_by_method = results_df[metrics].groupby(level=['name']).agg(_describe)
    print(stats_by_layer)
    print(stats_by_method)
    return results_df
        
    
def _run_single_eval(params):
    sample, reconstruct_func = params
    p_ij = reconstruct_func(sample)
    res = evaluate_reconstruction(sample, p_ij)
    return res


def _describe(data):
    return '{:.2f}±{:.2f}'.format(data.mean(), data.std())


def demo_random_single_layer():
    # tiny: layer_id=288
    # small: layer_id=202
    sample = fao_layer_sample()
    sample.print_summary()
    n = sample.n
    
    predictions = OrderedDict()
    eval_res = OrderedDict()
    experiments = [
        ('Random', random_baseline.reconstruct_layer_sample),
        ('IPF', ipf.reconstruct_layer_sample),
        ('DBCM', dbcm.reconstruct_layer_sample),
    ]
    
    with tqdm(experiments, desc='Reconstruction experiments') as experiments_pbar:
        for name, reconstruction_func in experiments_pbar:
            experiments_pbar.set_postfix_str(name)
            p_ij = reconstruction_func(sample)
            eval_res[name] = evaluate_reconstruction(sample, p_ij)
            predictions[name] = p_ij
    
    eval_res = pd.DataFrame.from_dict(eval_res, orient='index')
    print(eval_res)
    
    # Visualisation
    node_index = index_elements(sample.full.nodes)
    target_w = edges_to_matrix(sample.full.edges, node_index, node_index)
    demo_sample = np.empty(target_w.shape, dtype=np.int8)
    target_w.todense(out=demo_sample)
    
    observed_entries = matrix_intersetions(sample.observed.nodes, sample.node_index)
    observed_mask = np.ones_like(demo_sample)
    observed_mask[observed_entries] = 2
    demo_sample = (demo_sample * observed_mask).astype(np.float32) / 2
    
    res = [
        ('Source\n(observed - yellow,\nhidden - pink)', demo_sample),
        ('IPF', predictions['IPF']),
        ('DBCM', predictions['DBCM']),
        ('Random', predictions['Random']),
    ]
    fig = plt.figure(figsize=(8, 2.8))
    fig.suptitle(f'Reconstruction results – probability matrices (layer {sample.layer_id})')
    plt.axis('off')
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, len(res)),
                    cbar_mode='single',
                    label_mode=1)
    for i, ((name, mat), ax) in enumerate(zip(res, grid)):
        im = ax.imshow(mat,
                       cmap='inferno',
                       norm=mpl.colors.Normalize(0, 1, clip=True))
        ax.set_title(highlight_first_line(name),
                     fontsize=9,
                     verticalalignment='top',
                     y=-0.15)
        ax.set_axis_off()
        if i == 0:
            grid.cbar_axes[0].colorbar(im)
    plt.savefig('output/fao_reconst_comparison.png', dpi=300)
    plt.savefig('output/fao_reconst_comparison.svg')
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    res = demo_evaluate_all_layers()    
