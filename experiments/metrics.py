from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import toolz

from sampling import LayerSplit
from utils import display, matrix_intersetions, probabilies_to_adjacency

Edge = Tuple[int, int]


def binary_classification_metrics(target: Iterable[Edge], pred: Iterable[Edge]):
    target = set(target)
    pred = set(pred)
    num_expected = len(target)
    num_predicted = len(pred)
    
    tp = len(target & pred)
    fp = len(pred - target)
    fn = len(target - pred)
    # tn = n * n - (tp + fp + fn)
    
    if num_predicted != 0:
        precision = tp / (tp + fp)
    else:
        precision = 1
        
    if num_expected != 0:
        recall = tp / (tp + fn)
    else:
        recall = 1

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return dict(precision=precision, recall=recall, f1=f1,
                num_expected=num_expected, num_predicted=num_predicted)


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
        display(pd.Series(metrics))
    return metrics
