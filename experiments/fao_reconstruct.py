import random

import numpy as np
import pandas as pd
import toolz
from sklearn.model_selection import train_test_split

from sampling import GraphData, LayerSplit
from reconstruction import ipf, dbcm, random_baseline
from experiments.metrics import binary_classification_metrics
from fao_data import load_dataset
from utils import filter_by_layer, node_set


def random_layer_sample(layer_id=None, hidden_ratio=0.25, random_state=None) -> LayerSplit:
    """
    Select random layer, then split its nodes into 'observed' and 'hidden' parts.
    Edges that are adjacent to any hidden node are also considered as 'hidden'.

    Args:
        layer_id (int, optional): Chosen at random by default.
        hidden_ratio (float): Ratio of nodes hidden (0.25 by default).
                              Ratio of hidden edges is essentially larger.
        random_state (int): Random seed for splitting nodes.

    Returns:
        LayerSplit: data class containing nodes and edges for each partition:
                    'observed', 'hidden' and 'full'.
    """
    dataset = load_dataset()
    if layer_id is None:
        layer_id = random.choice(dataset.layer_names.index)
    
    edges = filter_by_layer(dataset.edges, layer_id)
    nodes = list(node_set(edges))
    nodes_observed, nodes_hidden = train_test_split(
        nodes,
        test_size=hidden_ratio,
        random_state=random_state
    )
    is_hidden_edge = edges.node_1.isin(nodes_hidden) | edges.node_2.isin(nodes_hidden)
    edges_hidden = edges[is_hidden_edge]
    edges_observed = edges[~is_hidden_edge]
    
    node_index = {node_id: i for i, node_id in enumerate(nodes)}
    return LayerSplit(
        layer_id=layer_id,
        node_index=node_index,
        observed=GraphData(edges=edges_observed, nodes=nodes_observed),
        hidden=GraphData(edges=edges_hidden, nodes=nodes_hidden),
        full=GraphData(edges=edges, nodes=nodes)
    )


def evaluate_reconstruction(
        layer_split: LayerSplit,
        probability_matrix: np.ndarray,
        silent: bool = False):
    n = len(layer_split.node_index)
    assert probability_matrix.shape == (n, n)
    
    # Target (hidden) edges. Converting to zero-based index.
    target_edges_src = toolz.get(layer_split.hidden.edges.node_1.tolist(), layer_split.node_index)
    target_edges_dst = toolz.get(layer_split.hidden.edges.node_2.tolist(), layer_split.node_index)
    target_edges = list(zip(target_edges_src, target_edges_dst))
    
    # Assigning zero probabilities to edges between observed nodes
    observed_node_ids = toolz.get(layer_split.observed.nodes, layer_split.node_index)
    probability_matrix[np.ix_(observed_node_ids, observed_node_ids)] = 0
    
    # Transforming probabilities into adjacency matrix and edge list
    pred_matrix = np.random.rand(n, n) < probability_matrix
    pred_edges_src, pred_edges_dst = pred_matrix.nonzero()
    pred_edges = list(zip(pred_edges_src, pred_edges_dst))
    
    metrics = binary_classification_metrics(target_edges, pred_edges)
    if not silent:
        print(pd.Series(metrics))
    return metrics


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    sample = random_layer_sample()
    sample.print_summary()
    n = sample.n
    
    print('IPF')
    ipf_w = ipf.reconstruct_layer_sample(sample)
    evaluate_reconstruction(sample, ipf_w)
    
    print('DBCM')
    dbcm_w = dbcm.reconstruct_layer_sample(sample)
    evaluate_reconstruction(sample, dbcm_w)
    
    print('RANDOM')
    random_w = random_baseline.reconstruct_layer_sample(sample)
    evaluate_reconstruction(sample, random_w)
