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
from sympy import Symbol
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
        
    # Solving eqn. (46) for z
    n = sample.n
    s_total = s_in.sum()
    z = Symbol('z')
    numerators = [z * s_out[i] * s_in[j] if i!=j else z * 0
                  for i in range(n)
                  for j in range(n)]
    p_ij = [num / (1 + num) for num in numerators]
    rhs = sum(tqdm(p_ij, desc='Summing p_ij', leave=False))
    lhs = s_total
    
    with tqdm(total=3, desc='Solving eqn for z', unit='step') as pbar:
        pbar.update()
        pbar.set_postfix_str('taking primes')
        rhs_prime = rhs.diff(z)
        rhs_prime2 = rhs_prime.diff(z)
        
        def f(x):
            return float(lhs - rhs.subs(z, x))
        
        def f_prime(x):
            return float(-rhs_prime.subs(z, x))
        
        def f_prime2(x):
            return float(-rhs_prime2.subs(z, x))
        
        pbar.update()
        for x0 in np.logspace(-5, 1):
            pbar.set_postfix_str(f'trying x0={x0}')
            sol = root_scalar(f, x0=x0, fprime=f_prime, fprime2=f_prime2, method='halley', xtol=1e-10)
            if sol.converged and sol.root > 0:
                break               
        pbar.update()
        
    logging.debug(sol)
    if not sol.converged:
        raise ValueError('Could not solve eqn for z')
    
    p_ij = [p.subs(z, sol.root) for p in p_ij]
    p_ij = np.array(p_ij).reshape(n, n)
    return p_ij
