"""
Module for splitting a graph layer into 'obesrved' and 'hidden' parts
"""
from typing import Dict, List, NamedTuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import NodeLabel, display, node_set


class GraphData(NamedTuple):
    edges: pd.DataFrame
    nodes: List[NodeLabel]


class LayerSplit(NamedTuple):
    layer_id: int
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
            }, orient='index')
        summary['obs.ratio'] = summary.observed / summary.total
        
        display(f'Summary of random split. Layer id: {self.layer_id}')
        display(summary)
    
    @property
    def n(self):
        return len(self.node_index)


def random_layer_split(edges: pd.DataFrame,
                       layer_id: int,
                       hidden_ratio: float = 0.5,
                       random_state: Optional[int] = None):
    nodes = list(node_set(edges))
    nodes_observed, nodes_hidden = train_test_split(
        nodes,
        test_size=hidden_ratio,
        random_state=random_state
    )
    
    edges_observed, edges_hidden = partition_into_observed_and_hidden(edges, nodes_hidden)
    node_index = {node_id: i for i, node_id in enumerate(nodes)}
    return LayerSplit(
        layer_id=layer_id,
        node_index=node_index,
        observed=GraphData(edges=edges_observed, nodes=nodes_observed),
        hidden=GraphData(edges=edges_hidden, nodes=nodes_hidden),
        full=GraphData(edges=edges, nodes=nodes)
    )
    

def layer_split_with_no_observables(edges: pd.DataFrame, layer_id: int):
    nodes = list(node_set(edges))
    node_index = {node_id: i for i, node_id in enumerate(nodes)}
    graph_data = GraphData(edges=edges, nodes=nodes)
    empty_graph = GraphData(
        edges=pd.DataFrame(columns=edges.columns, index=[]),
        nodes=[])
    return LayerSplit(
        layer_id=layer_id,
        node_index=node_index,
        observed=empty_graph,
        hidden=graph_data,
        full=graph_data
    )
    

def partition_into_observed_and_hidden(edges, nodes_hidden):
    is_hidden_edge = edges.node_1.isin(nodes_hidden) | edges.node_2.isin(nodes_hidden)
    edges_hidden = edges[is_hidden_edge]
    edges_observed = edges[~is_hidden_edge]
    return edges_observed, edges_hidden
