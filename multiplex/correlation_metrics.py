"""
Multiplexity and multireciprocity measures as described in:
[1] Gemmetto and Garlaschelli, ‘Reconstruction of Multiplex Networks with Correlated Layers’.

"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

from multiplex.multilayer_sampling import MultiLayerSplit


def count_common_links(edges: pd.DataFrame,
                       layer_a: int,
                       layer_b: int,
                       reciprocal: bool = False) -> int:
    cols = ['node_1', 'node_2']
    e_a = edges[edges.layer_id == layer_a][cols]
    e_b = edges[edges.layer_id == layer_b][cols]
    if reciprocal:
        e_b = e_b.rename(columns={'node_1': 'node_2', 'node_2': 'node_1'})
        
    num_common = len(
        e_a.set_index(cols).index.intersection(
        e_b.set_index(cols).index))
    return num_common


def multiplexity(edges: pd.DataFrame,
                 layer_a: int,
                 layer_b: int,
                 reciprocal: bool = False):
    num_common = count_common_links(edges, layer_a, layer_b, reciprocal=reciprocal)
    L_a = len(edges[edges.layer_id == layer_a])
    L_b = len(edges[edges.layer_id == layer_b])
    return 2 * num_common / (L_a + L_b)


def multiplexity_eval_metrics(sample: MultiLayerSplit,
                              pred_edges: pd.DataFrame) -> Dict[str, float]:
    assert len(sample.layer_ids) == 2, 'binary case only yet'
    layer_a, layer_b = sample.layer_ids
    
    orig_mltplx = multiplexity(sample.full.edges, layer_a, layer_b, reciprocal=False)
    orig_mltrcp = multiplexity(sample.full.edges, layer_a, layer_b, reciprocal=True)
    pred_mltplx = multiplexity(pred_edges, layer_a, layer_b, reciprocal=False)
    pred_mltrcp = multiplexity(pred_edges, layer_a, layer_b, reciprocal=True)
    
    res = dict(
        mltplx_tgt=orig_mltplx,
        mktplx_pred=pred_mltplx,
        mltplx_mae=mean_absolute_error([orig_mltplx], [pred_mltplx]),
        mltplx_mape=mean_absolute_percentage_error([orig_mltplx], [pred_mltplx]),
        mltrcp_mae=mean_absolute_error([orig_mltrcp], [pred_mltrcp]),
        mltrcp_mape=mean_absolute_percentage_error([orig_mltrcp], [pred_mltrcp])        
    )
    return res


if __name__ == '__main__':
    from fao_data import load_dataset
    
    dataset = load_dataset()
    N = 10
    layer_subset = dataset.layer_names.index[:N]
    common = np.zeros(shape=(N, N))
    mltplx = np.zeros(shape=(N, N))
    mltrcp = np.zeros(shape=(N, N))
    
    for i, layer_a in enumerate(layer_subset):
        for j, layer_b in enumerate(layer_subset):
            common[i, j] = count_common_links(dataset.edges, layer_a, layer_b, False)
            mltplx[i, j] = multiplexity(dataset.edges, layer_a, layer_b, False)
            mltrcp[i, j] = multiplexity(dataset.edges, layer_a, layer_b, True)
    
    print(common)
    print(mltplx.round(2))
    print(mltrcp.round(2))
