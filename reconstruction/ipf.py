"""
Reconstruction using Iterative Proportional Filling algorithm.

Target: single-layered network
Constraints: node strengths


[1] T. Squartini, G. Caldarelli, G. Cimini, A. Gabrielli, and D. Garlaschelli,
‘Reconstruction methods for networks: the case of economic and financial systems’,
Physics Reports, vol. 757, pp. 1–47, Oct. 2018, doi: 10.1016/j.physrep.2018.06.008.
"""

import numpy as np

from ops import repeat_col, repeat_row


def reconstruct(W, s_in, s_out, target_nodes=None, num_steps=3):
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
    
    for _ in range(num_steps):
        # STEP 1 - s_in. See eqn (15).
        _s_in = W_n.sum(axis=0)
        W_n = repeat_row(s_in, N) * (W_n / repeat_row(_s_in, N))
        
        # STEP 2 - s_out.
        _s_out = W_n.sum(axis=1)
        W_n = repeat_col(s_out, N) * (W_n / repeat_col(_s_out, N))
        
    return W_n
        


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
