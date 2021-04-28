from typing import Iterable, Optional

import pandas as pd
import networkx as nx
from networkx.algorithms.cluster import average_clustering

import toolz
from tqdm import tqdm


def edges_to_nx(edges: pd.DataFrame, nodes: Optional[Iterable] = None) -> nx.DiGraph:
    """
    Convert layers to networkx directed graph.
    If nodes are provided, the graph will contain all of them.

    >>> from fao_data import load_layers
    >>> edges = load_layers(1)
    >>> G = edges_to_nx(edges)
    >>> len(G.nodes)
    179
    """
    G = nx.DiGraph()
    if nodes is not None:
        G.add_nodes_from(nodes)

    if 'weight' in edges.columns:
        G.add_weighted_edges_from(edges[['node_1', 'node_2', 'weight']].values)
    else:
        G.add_edges_from(edges[['node_1', 'node_2']].values)
    return G


def clust_coef_by_layer(edges: pd.DataFrame,
                        num_workers: int = -1):
    """
    Compute average clustering coefficients for every layer

    Args:
        edges (pd.DataFrame)
        parallelize (bool): whether to enable process parallelization.

    Returns:
        clustering_coefficients(dict): dictionary like [layer_id -> coefficient]
        
    >>> from fao_data import load_layers
    >>> edges = load_layers([1,2])
    >>> clust_coef_by_layer(edges)
    {1: 0.49..., 2: 0.31...}
    >>> clust_coef_by_layer(edges, parallelize=True)
    {1: 0.49..., 2: 0.31...}
    """
    layer_ids, layer_edges = zip(*edges.groupby('layer_id'))
    
    get_cc_from_edges = toolz.compose(average_clustering,
                                      nx.DiGraph.to_undirected,
                                      edges_to_nx)

    if num_workers > 0:
        from tqdm.contrib.concurrent import process_map
        
        layer_cc = process_map(
            get_cc_from_edges,
            layer_edges,
            max_workers=num_workers,
            chunksize=8,
            desc='Clustering coefficients',
            leave=False)
    else:
        layer_cc = list(map(
            get_cc_from_edges,
            tqdm(layer_edges, desc='Clustering coefficients', leave=False)
        ))
    
    return dict(zip(layer_ids, layer_cc))


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
