"""
Reconstruction using Iterative Proportional Filling algorithm.

Target: single-layered network
Constraints: node strengths


[1] T. Squartini, G. Caldarelli, G. Cimini, A. Gabrielli, and D. Garlaschelli,
‘Reconstruction methods for networks: the case of economic and financial systems’,
Physics Reports, vol. 757, pp. 1–47, Oct. 2018, doi: 10.1016/j.physrep.2018.06.008.
"""

import logging

import numpy as np
import toolz
from tqdm import trange

from sampling import LayerSplit
from utils import index_elements, repeat_col, repeat_row


def reconstruct(W, s_in, s_out, max_steps=20, tol=1e-8) -> np.ndarray:
    """
    Run reconstrution for all or selected nodes, and return a modified weight matrix.
    See chapter 3.1.2 in [1].

    Args:
        W (2d array): weight matrix
        s_in, s_out (1d array): expected node strengths (sum of weights)
        target_nodes (iterable or none): which nodes to reconstruct (all by default)
    """
    N = len(s_in)
    W_n = W.copy()
    np.fill_diagonal(W_n, 0)
    
    steps_pbar = trange(max_steps, desc='IPF steps', leave=False)
    for step in steps_pbar:
        W_prev = W_n.copy()
        
        # STEP 1 - s_in. See eqn (15).
        _s_in = W_n.sum(axis=0)
        # Replacing some values with eps to avoid zero by zero division
        _s_in = np.where(s_in > 0, _s_in, 1e-10)
        W_n = repeat_row(s_in, N) * (W_n / repeat_row(_s_in, N))
        
        # STEP 2 - s_out.
        _s_out = W_n.sum(axis=1)
        _s_out = np.where(s_out > 0, _s_out, 1e-10)
        W_n = repeat_col(s_out, N) * (W_n / repeat_col(_s_out, N))
        
        if np.allclose(W_n, W_prev, atol=tol):
            steps_pbar.close()
            logging.debug(f'IPF converged in {step + 1} steps')
            break
    return W_n


def reconstruct_layer_sample(
        sample: LayerSplit,
        marginalized: bool = True,
        s_in: np.ndarray = None,
        s_out: np.ndarray = None) -> np.ndarray:
    """
    Run reconstruction routine given an experimantal sample.
    It is necessary for the IPF to initialize the weights
    with non-zero elements.

    Args:
        sample (LayerSplit): graph layer, split into observable and hidden parts.
        marginalized (bool): whether to use 'weight' column.
        s_in, s_out (1d array): expected node strengths.
            By default, this is estimated from the full layer data.

    Returns:
        predictions matrix (N⨯N numpy array)
    """
    nodes = sample.full.nodes
    
    if marginalized:
        strength_func = len
        weight_func = lambda _: 1
    else:
        strength_func = lambda edge_subset: edge_subset['weight'].sum()
        weight_func = lambda e: e.weight
        
    if s_out is None:
        s_out = sample.full.edges.groupby('node_1').apply(strength_func)
        s_out = np.array([s_out.get(u, 0) for u in nodes])
    if s_in is None:
        s_in = sample.full.edges.groupby('node_2').apply(strength_func)
        s_in = np.array([s_in.get(v, 0) for v in nodes])
    
    # IPF requires initial matrix to contain non-zero elements in place of
    # all entries where edges should be predicted. So, we use MaxEnt
    # matrix as an initial solution, and insert the observed part.
    W0 = _W_me(s_in, s_out)

    node_index = index_elements(nodes)
    observed_node_ids = toolz.get(sample.observed.nodes, node_index)
    W0[np.ix_(observed_node_ids, observed_node_ids)] = 0
    
    for edge in sample.observed.edges.itertuples():
        u = node_index[edge.node_1]
        v = node_index[edge.node_2]
        W0[(u, v)] = weight_func(edge)
    
    Wn = reconstruct(W0, s_in, s_out)
    return Wn
    

def _W_me(s_in, s_out):
    total_w = s_in.sum()
    s_out = s_out.reshape(-1, 1)
    s_in = s_in.reshape(1, -1)
    w_me = (s_out @ s_in) / total_w
    return w_me
    
    
if __name__ == '__main__':
    w = np.zeros((3,3))
    s_in = np.array([1, 2, 1])
    s_out = np.array([2, 1, 1])
    w_me = _W_me(s_in, s_out)
    print('MaxEnt matrix')
    print(w_me)
    
    w_0 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    w_ipf = reconstruct(w_0, s_in, s_out)
    print('Given matrix')
    print(w_0)
    print('IPF Method')
    print(w_ipf)
