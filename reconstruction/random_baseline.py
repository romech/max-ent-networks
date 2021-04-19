import numpy as np

from sampling import LayerSplit


def reconstruct_layer_sample(sample: LayerSplit) -> np.ndarray:
    n = len(sample.full.nodes)
    n_obs = len(sample.observed.nodes)
    n_hid = len(sample.hidden.nodes)
    m_hid = len(sample.hidden.edges)
    
    num_unobserved_entries = n * n  - n_obs * n_obs - n_hid
    unobserved_density = m_hid / num_unobserved_entries
    
    predicted_matrix = np.random.rand(n, n) * unobserved_density * 2
    np.fill_diagonal(predicted_matrix, 0)
    return predicted_matrix
