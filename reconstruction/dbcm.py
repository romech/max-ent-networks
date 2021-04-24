"""
Reconstruction using the Directed Binary Configuration Model.
See chapter 3.3.3 in [1].

Target: single-layered network
Constraints: node degrees


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
from utils import empirical_strengths, sparse_pairwise_product_matrix


def reconstruct_layer_sample(
        sample: LayerSplit,
        s_in: np.ndarray = None,
        s_out: np.ndarray = None,
        sol_order: int = 1) -> np.ndarray:
    
    if s_in is None:
        s_in, s_out = empirical_strengths(
            sample.full.edges,
            sample.full.nodes,
            marginalized=True
        )
    n = sample.n
        
    # Step 1 - Solving eqn. (46) for z
    with tqdm(total=3, desc='Solving eqn for z', unit='step', smoothing=1, leave=False) as pbar:
        pbar.set_postfix_str('preparing equation')
        
        # The equation in solved numerically, but symbolic calculations
        # are used for finding derivatives.
        z = Symbol('z')
        lhs = s_in.sum()
        
        # Let S(i,j) = s_out(i) * s_in(j)
        S = sparse_pairwise_product_matrix(s_out, s_in)
        p_ij = [S[i, j] / (1 / z + S[i, j])
                if i != j else z * 0
                for i in range(n)
                for j in range(n)]
        
        pbar.update()
        pbar.set_postfix_str('taking primes')
        f, *primes = _as_function_with_primes(z, lhs, p_ij, order=sol_order)
        
        pbar.update()
        # Solving numerically requires x0, so we try different values, just in case.
        for x0 in np.logspace(-5, 1, 10):
            pbar.set_postfix_str(f'trying x0={x0:.1e}')
            sol = root_scalar(
                f=f,
                fprime=primes[0] if sol_order > 0 else None,
                fprime2=primes[1] if sol_order > 1 else None,
                x0=x0,
                x1=1 if sol_order == 0 else None,
                xtol=1e-10)
            if sol.converged and sol.root > 0:
                break
        pbar.update()
        
    logging.debug(sol)
    if not sol.converged:
        raise ValueError('Could not solve eqn for z')
    
    # Step 2 - Compute p_ij values
    p_ij = [p.evalf(subs={z: sol.root}) for p in p_ij]
    p_ij = np.array(p_ij, dtype=np.float32).reshape(n, n)
    return p_ij


def _as_function_with_primes(z, lhs, p_ij, order=1):
    """
    lhs = const
    rhs = sum(p_ij(z))
    
    Returns:
        F(z)   = lhs - rhs(z) = 0
        F'(z)  = -rhs'(z)
        F''(z) = -rhs''(z)
    """
    rhs = Add(*p_ij, evaluate=False)
    
    @recover_from_nan
    def f(x):
        return float(lhs - rhs.subs(z, x))
    
    if order == 0:
        return (f,)
    
    # 1st order
    p_ij_pr = [p.diff(z) for p in p_ij]
    rhs_prime = Add(*p_ij_pr, evaluate=False)
    
    @recover_from_nan
    def fprime(x):
        return float(-rhs_prime.subs(z, x))
    
    if order == 1:
        return f, fprime
    
    # 2nd order
    p_ij_pr2 = [p_pr.diff(z) for p_pr in p_ij_pr]
    rhs_prime2 = Add(*p_ij_pr2, evaluate=False)
    
    @recover_from_nan
    def fprime2(x):
        return float(-rhs_prime2.subs(z, x))
    
    return f, fprime, fprime2


def _nan_fallback(_):
    return np.nan
    

def recover_from_nan(f):
    return excepts(TypeError, f, _nan_fallback)
