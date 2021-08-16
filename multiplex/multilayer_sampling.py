"""
Module for splitting a graph layer into 'obesrved' and 'hidden' parts
"""
from functools import reduce
from operator import and_
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

from sampling import partition_into_observed_and_hidden, GraphData, LayerSplit
from utils import display, node_set, filter_by_layer, edges_to_matrix, index_elements, NodeLabel


class MultiLayerSplit(NamedTuple):
    layer_ids: List[int]
    node_index: Dict[NodeLabel, int]
    observed: GraphData
    hidden: GraphData
    full: GraphData
    
    def print_summary(self):
        # TODO: display number of common and unique nodes
        layer_splits = [self.select_single(layer_id, all_nodes=False)
                        for layer_id in self.layer_ids]
        # num_common_nodes = {u for layer_split in layer_splits for u in layer_split.node_index.keys()}
        node_sets = (layer_split.node_index.keys() for layer_split in layer_splits)
        num_common_nodes = len(reduce(and_, node_sets))
        display(f'PRINTING SUMMARY FOR LAYERS {self.layer_ids}')
        display('Total {} nodes, {} are common.'.format(len(self.node_index), num_common_nodes))
        
        for layer_split in layer_splits:
            display('---------')
            layer_split.print_summary()
    
    @property
    def n(self):
        return len(self.node_index)
    
    def select_single(self, layer_id: int, all_nodes: bool) -> LayerSplit:
        e_obs = filter_by_layer(self.observed.edges, layer_id)
        e_hid = filter_by_layer(self.hidden.edges, layer_id)
        e_full = filter_by_layer(self.full.edges, layer_id)
        if all_nodes:
            v_obs = self.observed.nodes
            v_hid = self.hidden.nodes
            v_full = self.full.nodes
            node_index = self.node_index
        else:
            v_obs = sorted(node_set(e_obs))
            v_hid = sorted(node_set(e_hid))
            v_full = sorted(node_set(e_full))
            node_index = index_elements(v_full)
        
        return LayerSplit(
            layer_id=layer_id,
            node_index=node_index,
            observed=GraphData(edges=e_obs, nodes=v_obs),
            hidden=GraphData(edges=e_hid, nodes=v_hid),
            full=GraphData(edges=e_full, nodes=v_full),
        )
    

def multilayer_sample(edges: pd.DataFrame,
                      layer_ids: List[int],
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
    nodes_observed = nodes_observed.flatten().tolist()
    nodes_hidden = nodes_hidden.flatten().tolist()
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
