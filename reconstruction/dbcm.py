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
from scipy.optimize import root_scalar
from sympy import Add, Symbol
from toolz import excepts
from tqdm import tqdm

from sampling import LayerSplit
from utils import empirical_strengths


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
    n = sample.n
        
    # Solving eqn. (46) for z
    with tqdm(total=3, desc='Solving eqn for z', unit='step') as pbar:
        pbar.set_postfix_str('preparing equation')
        z = Symbol('z')
        lhs = s_in.sum()
        numerators = [z * s_out[i] * s_in[j] if i!=j else z * 0
                      for i in range(n)
                      for j in range(n)]
        p_ij = [num / (1 + num) for num in numerators]
        rhs = Add(*p_ij)
        
        pbar.update()
        pbar.set_postfix_str('taking primes')
        p_ij_pr = [p.diff(z) for p in p_ij]
        p_ij_pr2 = [p_pr.diff(z) for p_pr in p_ij_pr]
        rhs_prime = Add(*p_ij_pr)
        rhs_prime2 = Add(*p_ij_pr2)
        
        def f(x):
            return float(lhs - rhs.subs(z, x))
        
        def f_prime(x):
            return float(-rhs_prime.subs(z, x))
        
        def f_prime2(x):
            return float(-rhs_prime2.subs(z, x))
        
        def nan_fallback(_):
            return np.nan
        
        pbar.update()
        for x0 in np.logspace(-5, 1, 10):
            pbar.set_postfix_str(f'trying x0={x0:.1e}')
            sol = root_scalar(
                excepts(TypeError, f, nan_fallback),
                fprime=excepts(TypeError, f_prime, nan_fallback),
                fprime2=excepts(TypeError, f_prime2, nan_fallback),
                x0=x0,
                method='halley',
                xtol=1e-10)
            if sol.converged and sol.root > 0:
                break
        pbar.update()
        
    logging.debug(sol)
    if not sol.converged:
        raise ValueError('Could not solve eqn for z')
    
    p_ij = [p.subs(z, sol.root) for p in p_ij]
    p_ij = np.array(p_ij, dtype=np.float32).reshape(n, n)
    return p_ij
