import itertools
from typing import Tuple, List

import numpy as np
import toolz
from tqdm.contrib.concurrent import process_map


def multiplexity_score(weights_i, weights_j, W_i, W_j):
    """
    Args:
    - weights_i, weights_j:
        ┌───────┬────────┐
        │ index │ weight │
        ├───┬───┼────────┤
        │ 2 │ 1 │   42   │
        │ 4 │ 9 │   10   │
        │ … │ … │   ……   │
        └───┴───┴────────┘
    
    - W_i, W_j: total weight of edges
    
    """

    common_edges = weights_i.join(weights_j, how='inner', lsuffix='_i', rsuffix='_j')
    if len(common_edges) == 0:
        return 0
    min_weight = np.min(common_edges.values, axis=1)
    multiplexity = 2 * min_weight.sum() / (W_i + W_j)
    return multiplexity


def multiplexity_score_from_tuple(args):
    return multiplexity_score(*args)


def pairwise_multiplexity(edges, marginalized=True, rescaled=False) -> Tuple[np.ndarray, List]:
    """
    Args:
    - edges: DataFrame with columns
        * layer_id
        * node_1
        * node_2
        * weight
    - marginalized: whether to use adjacency of weights
    
    Returns:
    - multiplexity_matrix: numpy matrix of shape (num_layers, num_layers)
    - layer_ids: array of the layer indices (values from `layer_id` column)
                 in the order they appean in matrix
    """
    if marginalized:
        edges = edges.copy()
        edges['weight'] = edges['weight'].apply(lambda x: x > 0)
        
    layers = edges.groupby('layer_id')
    num_layers = len(layers)
    W = layers.apply(lambda e: e.weight.sum())
    
    layer_ids = [idx for idx, _ in layers]
    layer_weights = {idx: edges_i.set_index(['node_1', 'node_2']).drop(columns='layer_id')
                     for idx, edges_i in layers}
    
    ii = list(range(num_layers))
    idx_ii = toolz.get(ii, layer_ids)
    idx_ij = list(itertools.product(idx_ii, repeat=2))
    params = [
        (layer_weights[idx_i], layer_weights[idx_j], W[idx_i], W[idx_j])
        for idx_i, idx_j in idx_ij
    ]
    
    res_list = process_map(
        multiplexity_score_from_tuple,
        params,
        max_workers=6,
        chunksize=64,
        desc='Pairwise multiplexity',
        unit='pair'
    )
    multiplexity_matrix = np.array(res_list).reshape(num_layers, num_layers)
    return multiplexity_matrix, layer_ids
