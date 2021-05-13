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
import warnings

import numpy as np
from scipy.optimize import root_scalar
from sympy import Add, Symbol
from toolz import curry, excepts
from tqdm.auto import tqdm

from sampling import LayerSplit
from utils import empirical_strengths, sparse_pairwise_product_matrix


@curry
def reconstruct_layer_sample(
        sample: LayerSplit,
        s_in: np.ndarray = None,
        s_out: np.ndarray = None,
        enforce_observed: bool = True,
        sol_order: int = 1) -> np.ndarray:
    
    if s_in is None:
        s_in, s_out = empirical_strengths(
            sample.full.edges,
            sample.full.nodes,
            marginalized=True
        )
    n = sample.n
    get_observed_value = _get_oserved_value_func(sample)
    
    # Step 1 - Solving eqn. (46) for z
    with warnings.catch_warnings(), \
         tqdm(total=3, desc='Solving eqn for z',
              unit='step', smoothing=1, leave=False) as pbar:
             
        warnings.filterwarnings("ignore", message="Derivative was zero") 
        pbar.set_postfix_str('preparing equation')
        
        # The equation in solved numerically, but symbolic calculations
        # are used for finding derivatives.
        # Variable `z` here corresponds to 1/z in the original equation.
        z = Symbol('z')
        lhs = s_in.sum()
        
        # Let S(i,j) = s_out(i) * s_in(j)
        S = sparse_pairwise_product_matrix(s_out, s_in)
        
        def p_ij_expr(i, j):
            if i == j:
                return z * 0
            if enforce_observed:
                observed = get_observed_value(i, j)
                if observed is not None:
                    return float(observed) + z * 0
            return S[i, j] / (z + S[i, j])
            
        p_ij = [p_ij_expr(i, j)
                for i in range(n)
                for j in range(n)]
        
        pbar.update()
        pbar.set_postfix_str('taking primes')
        f_with_primes = _as_function_with_primes(z, lhs, p_ij, order=sol_order)
        
        pbar.update()
        x0_estimates = _get_x0_estimates(s_in, s_out)
        for trial, x0 in enumerate(x0_estimates, 1):
            pbar.set_postfix_str(f'trying x0={x0:.1e}')
            sol = root_scalar(
                f=f_with_primes,
                fprime=sol_order > 0,
                fprime2=sol_order > 1,
                x0=x0,
                x1=1 if sol_order == 0 else None,
                xtol=1e-10,
                maxiter=30)
            if sol.converged and sol.root > 0:
                x0_estimates = x0_estimates[:trial]
                break
        pbar.update()
        
    logging.debug(sol)
    logging.debug(f"Finished after trying x0={x0_estimates}")
    
    if not sol.converged:
        logging.warning(f'Could not solve eqn for z, layer_id={sample.layer_id}')
        return np.zeros((n, n))
    
    # Step 2 - Compute p_ij values
    p_ij = [p.evalf(subs={z: sol.root}) for p in p_ij]
    p_ij = np.array(p_ij, dtype=np.float32).reshape(n, n)
    return p_ij


def _get_oserved_value_func(sample):
    observed_nodes = set(sample.observed.nodes)
    observed_edges = sample.observed.edges.set_index(['node_1', 'node_2']).index
    
    def get_observed_value(i, j):
        idx_i = sample.full.nodes[i]
        idx_j = sample.full.nodes[j]
        if (idx_i in observed_nodes) and (idx_j in observed_nodes):
            return (idx_i, idx_j) in observed_edges
        else:
            return None
    return get_observed_value
    

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
        return float((lhs - rhs).evalf(subs={z: x}))

    if order == 0:
        return f
    
    # 1st order
    p_ij_pr = [p.diff(z) for p in p_ij]
    rhs_prime = Add(*p_ij_pr, evaluate=False)
    
    @recover_from_nan
    def f_df(x):
        df = float(-rhs_prime.evalf(subs={z: x}))
        return f(x), df
    
    if order == 1:
        return f_df
    
    # 2nd order
    p_ij_pr2 = [p_pr.diff(z) for p_pr in p_ij_pr]
    rhs_prime2 = Add(*p_ij_pr2, evaluate=False)
    
    @recover_from_nan
    def f_df_d2f(x):
        f, df = f_df(x)
        d2f = float(-rhs_prime2.evalf(subs={z: x}))
        return f, df, d2f
    
    return f_df_d2f


def _get_x0_estimates(s_in, s_out):
    # Unsure about this. But I think this interval is wide enough to find soultion
    x0_estimates = np.logspace(1, np.log10(s_in.max() * s_out.max()), 10)
    np.random.shuffle(x0_estimates)
    # The paper says that MECAPM uses z=1/|W|, so we give it a try first.
    return [s_in.sum()] + x0_estimates.tolist()

def _nan_fallback(_):
    return np.nan
    

def recover_from_nan(f):
    return excepts(TypeError, f, _nan_fallback)
