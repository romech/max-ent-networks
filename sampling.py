from typing import Dict, List, NamedTuple, Union

import pandas as pd


NodeLabel = Union[int, str]

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
        
        print('Summary of random split. Layer id:', self.layer_id)
        print(summary)
    
    @property
    def n(self):
        return len(self.node_index)
