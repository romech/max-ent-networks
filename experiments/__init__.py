from typing import Any, Dict, Hashable, List, Iterable, NamedTuple, Tuple, Union

import pandas as pd


Edge = Tuple[int, int]
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
    

def binary_classification_metrics(target: Iterable[Edge], pred: Iterable[Edge]):
    target = set(target)
    pred = set(pred)
    support = len(target)
    num_predicted = len(pred)
    
    tp = len(target & pred)
    fp = len(pred - target)
    fn = len(target - pred)
    # tn = n * n - (tp + fp + fn)
    
    if num_predicted != 0:
        precision = tp / (tp + fp)
    else:
        precision = 1
        
    if support != 0:
        recall = tp / (tp + fn)
    else:
        recall = 1

    f1 = 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f1=f1,
                support=support, num_predicted=num_predicted)

