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
    
    # Pm = P_a * P_b
    # fulfilled_common = Pm[Pm==1].sum()
    # expected_additionally = Pm[Pm<1].sum()
    # if np.abs(expected_additionally) < 1e-6:
    #     k = 1
    # elif expected_additionally < 0:
    #     logging.info('Unusual case: target number of common links '
    #                  'is less than predicted using baseline')
    #     k = target_common / Pm.sum()
    # else:
    #     k = (target_common - fulfilled_common) / expected_additionally
    Pm = P_a * P_b
    upper_bounds = np.minimum(P_a, P_b)
    if upper_bounds.sum() < target_common:
        logging.info('Target number of common links is unreachable')
    if target_common > 0:
        k = _find_opt_k(Pm, upper_bounds, target_common)
    else:
        k = 1
    logging.debug("Correction coefficient %.3f", k)
    p11 = np.clip(np.minimum(Pm * k, upper_bounds), 0, 1)
    p01 = P_b - p11
    p10 = P_a - p11
    p00 = np.clip(1 - (p01 + p10 + p11), 0, 1)
    assert all(map(verify_finite, (p00, p01, p10, p11)))
    
    if not np.allclose(p00 + p01 + p10 + p11, np.ones_like(p00), atol=1e-6):
        logging.warning("Probabilities don't add up to 1")
        
    adj_a, adj_b = sample_joint_proba(p00, p01, p10, p11)
    a_common = adj_a * adj_b
    
    logging.debug("Common links: ground truth %d, expected value %.1f, generated %d, k=%.2f",
                  target_common, p11.sum(), np.count_nonzero(a_common), k)
    return adj_a, adj_b


def sample_joint_proba(p00, p01, p10, p11) -> MatrixPair:
    n = p00.shape[0]
    adj_a = np.empty_like(p00, dtype=np.int8)
    adj_b = np.empty_like(p00, dtype=np.int8)
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i in range(n):
        for j in range(n):
            prob_1d = np.array([p00[i, j], p01[i, j], p10[i, j], p11[i, j]])
            adj_a[i, j], adj_b[i, j] = outcomes[np.random.choice(4, p=prob_1d)]
    return adj_a, adj_b


def _find_opt_k(Pm, upper_bounds, target_common, max_steps=10):
    k0 = target_common / Pm.sum()
    
    def f(k):
        expected_common = np.minimum(Pm * k, upper_bounds).sum()
        return target_common - expected_common
    
    sol = root_scalar(f, x0=k0, x1=k0*2, xtol=0.5, rtol=0.001, maxiter=max_steps)
    if sol.converged:
        return sol.root
    else:
        print(sol)
        return k0
