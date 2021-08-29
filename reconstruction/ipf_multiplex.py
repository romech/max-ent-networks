import logging
from typing import Tuple, Union

import numpy as np
from scipy.optimize import root_scalar

from reconstruction import ipf
from multiplex.correlation_metrics import count_common_links
from multiplex.multilayer_sampling import MultiLayerSplit
from utils import verify_finite


MatrixPair = Tuple[np.ndarray, np.ndarray]


def two_layer_ipf(sample: MultiLayerSplit,
                  ipf_steps: int = 20,
                  return_ipf_outputs: bool = False) -> Union[MatrixPair, Tuple[MatrixPair, MatrixPair]]:
    layer_a, layer_b = sample.layer_ids
    sample_a = sample.select_single(layer_a, all_nodes=True)
    sample_b = sample.select_single(layer_b, all_nodes=True)
    P_a = ipf.reconstruct_v2(sample_a, ipf_steps=ipf_steps)
    P_b = ipf.reconstruct_v2(sample_b, ipf_steps=ipf_steps)
    target_common = count_common_links(sample.full.edges,
                                       layer_a=layer_a,
                                       layer_b=layer_b,
                                       reciprocal=False)
    adj_a, adj_b = two_layer_multiplexity_tuning(P_a, P_b, target_common)
    
    if not return_ipf_outputs:
        return adj_a, adj_b
    else:
        return (adj_a, adj_b), (P_a, P_b)
        
        
def two_layer_multiplexity_tuning(P_a: np.ndarray,
                                  P_b: np.ndarray,
                                  target_common: int) -> MatrixPair:
    """
    Generate adjacency matrices from two probability matrices,
    so that the resulting number of links in common is equal to
    some expected value.

    Args:
        P_a, P_b: two NxN matrices of link probabilities
        target_common: expected number of common links

    Returns:
        (A_1, A_2): adjacency matrices
    """
    if not (verify_finite(P_a) and verify_finite(P_b)):
        raise ValueError('Probability matrices P_a, P_b contain unexpected values')
    
    Pm = P_a * P_b
    
    # Defining constraints for every kappa_ij.
    # Ignoring zero dvision because these values are replaced.
    with np.errstate(divide='ignore', invalid='ignore'):
        min_kappa = np.where(Pm > 0, (P_a + P_b - 1) / Pm, np.zeros_like(Pm))
        max_kappa = np.where(Pm > 0, np.minimum(1 / P_a, 1 / P_b), np.zeros_like(Pm))
    
    if target_common > 0:
        k = _find_opt_k(Pm, min_kappa, max_kappa, target_common)
    else:
        k = 1
    logging.debug("Correction coefficient %.3f", k)
    kappa_matrix = _get_kappa_matrix(k, min_kappa, max_kappa)
    p00, p01, p10, p11 = _get_rescaled_probabilities(P_a, P_b, Pm, kappa_matrix)
        
    adj_a, adj_b = _sample_joint_proba(p00, p01, p10, p11)
    a_common = adj_a * adj_b
    
    logging.debug("Common links: ground truth %d, expected value %.1f, generated %d, k=%.2f",
                  target_common, p11.sum(), np.count_nonzero(a_common), k)
    return adj_a, adj_b


def _get_rescaled_probabilities(P_a, P_b, Pm, kappa_matrix):
    """
    Compute joint probabilities for every possible outcome:
    p00 - no links between a pair of nodes
    p01 - only layer `b` contains a link
    p10 - only layer `a` contains a link
    p11 - both layers contain a link

    Args:
        P_a, P_b (NxN matrices): Probabilistic adjacency matrices
        Pm (NxN matrix): Basically P_a*P_b
        kappa_matrix (NxN matrix): Correction coefficient matrix

    Returns:
        p00, p01, p10, p11 (NxN matrices)
    """
    p11 = Pm * kappa_matrix
    p01 = P_b - p11
    p10 = P_a - p11
    p00 = 1 - (p01 + p10 + p11)
    
    # Due to numeric computations, some values exceed bounds a bit.
    # But large deviations might be caused by something else.
    assert (p00 >= -1e-10).all()
    assert (p01 >= -1e-10).all()
    assert (p10 >= -1e-10).all()
    assert all(map(verify_finite, (p00, p01, p10, p11))), 'nan or inf values are not ok'
    
    p01 = np.clip(p01, 0, 1)
    p10 = np.clip(p10, 0, 1)
    p00 = np.clip(p00, 0, 1)
    if not np.allclose(p00 + p01 + p10 + p11, 1, atol=1e-8):
        logging.warning("Probabilities don't add up to 1")
    return p00, p01, p10, p11


def _sample_joint_proba(p00, p01, p10, p11) -> MatrixPair:
    n = p00.shape[0]
    adj_a = np.empty_like(p00, dtype=np.int8)
    adj_b = np.empty_like(p00, dtype=np.int8)
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i in range(n):
        for j in range(n):
            prob_1d = np.array([p00[i, j], p01[i, j], p10[i, j], p11[i, j]])
            adj_a[i, j], adj_b[i, j] = outcomes[np.random.choice(4, p=prob_1d)]
    return adj_a, adj_b


def _find_opt_k(Pm, min_kappa, max_kappa, target_common, max_steps=20):
    k0 = target_common / Pm.sum()
    
    def absolute_error_in_mutual_links(k):
        rescaled_prob = Pm * _get_kappa_matrix(k, min_kappa, max_kappa)
        expected_common = rescaled_prob.sum()
        return target_common - expected_common
    
    max_k = max_kappa.max()
    y0 = absolute_error_in_mutual_links(k0)
    y1 = absolute_error_in_mutual_links(max_k)
    if y0 == 0:
        return k0
    if y1 == 0:
        return max_k
    if y1 > 0:
        logging.info('Target number of common links is unreachable')
        return max_k
    
    # Need to define such lower bound that function signs on bounds are opposite
    if np.sign(y0) * np.sign(y1) > 0:
        k0 = 0
    
    try:
        sol = root_scalar(
            absolute_error_in_mutual_links,
            x0=k0,
            bracket=(k0, max_k),
            xtol=0.5,
            rtol=0.001,
            maxiter=max_steps)
    except ValueError as e:
        logging.error('Caught exception while solving the equation for k: %s', e)
        return k0
        
    if sol.converged:
        return sol.root
    else:
        print(sol)
        logging.info("Didn't converge. Expected L error: %.2f",
                     absolute_error_in_mutual_links(sol.root))
        return sol.root


def _get_kappa_matrix(k, min_kappa, max_kappa):
    return np.clip(np.full_like(min_kappa, k), min_kappa, max_kappa)
