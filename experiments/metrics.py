from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import toolz
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

from sampling import LayerSplit
from utils import (display, empirical_strengths, matrix_intersetions,
                   probabilies_to_adjacency_advanced)

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


def check_constraints(sample, probability_matrix):
    s_in, s_out = empirical_strengths(sample.full.edges,
                                      sample.full.nodes,
                                      marginalized=True)
    """
    Compute deviation of s_in/s_out from expected values.
    Return dict with MAE and weighted MAPE.
    """
    _s_in = probability_matrix.sum(axis=1)
    _s_out = probability_matrix.sum(axis=0)
    corr_in, pv_in = pearsonr(s_in, _s_in)
    corr_out, pv_out = pearsonr(s_out, _s_out)
    spcorr_in, sp_pv_in = spearmanr(s_in, _s_in)
    spcorr_out, sp_pv_out = spearmanr(s_out, _s_out)
    return dict(
        s_in_mae=mean_absolute_error(s_in, _s_in),
        s_out_mae=mean_absolute_error(s_out, _s_out),
        s_in_mape=mean_absolute_percentage_error(s_in, _s_in),
        s_out_mape=mean_absolute_percentage_error(s_out, _s_out),
        # r2_in=r2_score(s_in, _s_in),
        # r2_out=r2_score(s_out, _s_out),
        s_in_js=jensenshannon(s_in, _s_in, base=2) ** 2,
        s_out_js=jensenshannon(s_out, _s_out, base=2) ** 2,
        corr_in=corr_in,
        corr_out=corr_out,
        spcorr_in=spcorr_in,
        spcorr_out=spcorr_out,
        pv_in=pv_in,
        pv_out=pv_out,
        sp_pv_in=sp_pv_in,
        sp_pv_out=sp_pv_out,
    )


def evaluate_reconstruction(
        sample: LayerSplit,
        probability_matrix: np.ndarray,
        verbose: bool = False):
    n = len(sample.node_index)
    assert probability_matrix.shape == (n, n)
    
    # First, check constraints
    constr_metrics = check_constraints(sample, probability_matrix)
    
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
    pred_matrix = probabilies_to_adjacency_advanced(probability_matrix)
    pred_edges_src, pred_edges_dst = pred_matrix.nonzero()
    pred_edges = list(zip(pred_edges_src, pred_edges_dst))
    
    metrics = binary_classification_metrics(target_edges, pred_edges)
    metrics.update(constr_metrics)
    if verbose:
        display(pd.Series(metrics))
    return metrics
