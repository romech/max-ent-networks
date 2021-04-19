"""
Reconstruction using the Directed Binary Configuration Model.
See chapter 3.3.3 in [1].

Target: single-layered network
Constraints: node strengths


[1] T. Squartini, G. Caldarelli, G. Cimini, A. Gabrielli, and D. Garlaschelli,
‘Reconstruction methods for networks: the case of economic and financial systems’,
Physics Reports, vol. 757, pp. 1–47, Oct. 2018, doi: 10.1016/j.physrep.2018.06.008.
"""
import logging

import numpy as np
import toolz
from sympy import Symbol
from sympy.solvers import solve, solveset, S
from tqdm import trange, tqdm

from sampling import LayerSplit
from utils import empirical_strengths, index_elements, repeat_col, repeat_row


def reconstruct_layer_sample(
        sample: LayerSplit,
        marginalized: bool = True,
        s_in: np.ndarray = None,
        s_out: np.ndarray = None) -> np.ndarray:
    
    if not marginalized:
        raise NotImplementedError()
    
    if s_in is None:
        s_in, s_out = empirical_strengths(
            sample.full.edges,
            sample.full.nodes,
            marginalized=marginalized
        )

    # Solving eqn. (46) for z
    # TAKES INFINITE TIME TO COMPLETE
    n = sample.n
    s_total = s_in.sum() + s_out.sum()
    z = Symbol('z')
    numerators = [z * s_out[i] * s_in[j] if i!=j else z * 0
                  for i in range(n)
                  for j in range(n)]
    p_ij = [num / (1 + num) for num in numerators]
    eqn = sum(tqdm(p_ij, desc='Summing p_ij')) - s_total
    with tqdm(total=1, desc='Solving eqn for z') as pbar:
        sol = solve(eqn, z, particular=True, quick=True)
        pbar.update(1)
        pbar.set_postfix({'z': str(sol)})
    
    print(sol)
    
    if len(sol) == 0:
        print('Empty solution')
        return
    p_ij = [p.subs(z, sol[0]) for p in p_ij]
    p_ij = np.array(p_ij).reshape(n, n)
    logging.debug(f'z = {sol[0]}')
    return p_ij
