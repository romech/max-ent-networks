"""
Module for splitting a graph layer into 'obesrved' and 'hidden' parts
"""
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

from sampling import partition_into_observed_and_hidden, GraphData, NodeLabel
from utils import display, node_set, filter_by_layer, edges_to_matrix, index_elements


class MultiLayerSplit(NamedTuple):
    layer_ids: List[int]
    node_index: Dict[NodeLabel, int]
    observed: GraphData
    hidden: GraphData
    full: GraphData
    
    def print_summary(self):
        summary = pd.DataFrame.from_dict({
                'nodes': {
                    'total': len(self.full.nodes),
                    'observed': len(self.observed.nodes),
                    'hidden': len(self.hidden.nodes),
                },
                'edges': {
                    'total': len(self.full.edges),
                    'observed': len(self.observed.edges),
                    'hidden': len(self.hidden.edges),
                }
            }, orient='index')\
            .append(pd.concat([
                self.full.edges.layer_id.value_counts().rename('total'),
                self.observed.edges.layer_id.value_counts().rename('observed'),
                self.hidden.edges.layer_id.value_counts().rename('hidden')],
                axis=1).rename('layer {}'.format, axis='index')
            )
        
        summary['obs.ratio'] = summary.observed / summary.total
        
        display(f'Summary of random split. Layer ids: {self.layer_ids}')
        display(summary)
    
    @property
    def n(self):
        return len(self.node_index)
    

def multilayer_sample(edges: pd.DataFrame,
                      layer_ids: List[NodeLabel],
                      hidden_ratio: float = 0.5,
                      random_state: Optional[int] = None) -> MultiLayerSplit:
    """
    Split multilayer network into hidden and observed parts
    for specified layers. First, split nodes at random, then
    split edges accordingly.
    
    Usage example:   
    
    >>> from fao_data import load_all_layers
    >>> edges = load_all_layers()
    >>> sample = multilayer_sample(edges, [42, 123], random_state=0)
    >>> sample.print_summary()
    Summary of random split. Layer ids: [42, 123]
               total  observed  hidden  obs.ratio
    nodes        136        67      69   0.492647
    edges       1034       248     786   0.239845
    layer 42     773       179     594   0.231565
    layer 123    261        69     192   0.264368
    """
    
    edges = filter_by_layer(edges, layer_ids)
    nodes = sorted(node_set(edges))
    node_layers = _node_layer_incidence(edges, nodes, layer_ids)
    np.random.seed(random_state)
    nodes_observed, _, nodes_hidden, _ = \
        iterative_train_test_split(np.array(nodes).reshape(-1, 1),
                                   node_layers,
                                   test_size=hidden_ratio)
    nodes_observed = nodes_observed.flatten()
    nodes_hidden = nodes_hidden.flatten()
    edges_observed, edges_hidden = partition_into_observed_and_hidden(edges, nodes_hidden)
    split = MultiLayerSplit(
        layer_ids=layer_ids,
        node_index=index_elements(nodes),
        observed=GraphData(edges_observed, nodes_observed),
        hidden=GraphData(edges_hidden, nodes_hidden),
        full=GraphData(edges, nodes)
    )
    return split
    

def _node_layer_incidence(edges, nodes, layer_ids) -> np.matrix:
    """
    Generate binary relation matrix where rows correspond to nodes,
    and columns to layers.
    
    >>> nodes = [123, 456, 789]
    >>> layer_ids = [0, 1]
    >>> edges = pd.DataFrame(
    ...     [[123, 456, 0],
    ...      [456, 789, 1]],
    ...     columns=('node_1', 'node_2', 'layer_id'))
    >>> _node_layer_incidence(edges, nodes, layer_ids)
    matrix([[1, 0],
            [1, 1],
            [0, 1]], dtype=int8)
    """
    incidence_df = \
        edges[['node_1', 'layer_id']]\
            .rename(columns={'node_1': 'node'})\
            .append(edges[['node_2', 'layer_id']]\
                    .rename(columns={'node_2': 'node'}))
    
    incidence_matrix = edges_to_matrix(
        incidence_df,
        src_index=index_elements(nodes),
        dst_index=index_elements(layer_ids),
        src_col='node',
        dst_col='layer_id'
    )
    return incidence_matrix.todense()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
