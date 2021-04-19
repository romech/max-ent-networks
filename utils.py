import traceback
from collections import abc
from itertools import chain
from typing import (Any, Dict, Hashable, Iterable, List, Mapping, Optional,
                    Set, Tuple, Union)

import numpy as np
import pandas as pd
import toolz
from scipy.sparse import csr_matrix


def replace_underscores(string):
    return string.replace('_', ' ')


def fallback(f):
    """
    Recover from any exception, print traceback, and return None.
    """
    
    return toolz.excepts(Exception, f, lambda _: traceback.print_exc())


def put_col_in_front(df, col):
    cols = list(df.columns)
    cols.remove(col)
    return df[[col] + cols]


def node_set(edges: pd.DataFrame,
             src_col: str = 'node_1',
             dst_col: str = 'node_2') -> Set[int]:
    
    unq_1 = edges[src_col].unique()
    unq_2 = edges[dst_col].unique()
    unk = set(chain(unq_1, unq_2))
    return unk


def node_set_size(edges: pd.DataFrame) -> int:
    return len(node_set(edges))


def repeat_col(array: np.ndarray, n: int):
    return array.reshape(-1, 1).repeat(n, axis=1)


def repeat_row(array: np.ndarray, n: int):
    return array.reshape(1, -1).repeat(n, axis=0)


def index_elements(elements: Iterable[Hashable]) -> Dict[Hashable, int]:
    return {elem: i for i, elem in enumerate(elements)}


def filter_by_layer(edges: pd.DataFrame, layer_ids: Union[int, str, List, Tuple]) -> pd.DataFrame:
    if isinstance(layer_ids, (str, int, np.integer)):
        crit = edges.layer_id == layer_ids
    elif isinstance(layer_ids, abc.Iterable):
        crit = edges.layer_id.isin(layer_ids)
    else:
        raise ValueError(f'Expected int/str/list for layer_ids parameter, got {type(layer_ids)}')
    return edges[crit]


def edges_to_matrix(edges: pd.DataFrame,
                    src_index: Mapping[Any, int],
                    dst_index: Mapping[Any, int],
                    src_col: str = 'node_1',
                    dst_col: str = 'node_2',
                    weight_col: Optional[str] = None) -> csr_matrix:
    """
    Create sparse adjaceny/weight matrix given edge DataFrame.

    Args:
        edges (DataFrame): edge list, either weighted or not.
        src_index, dst_index (dict-like): mapping from node indices to zero-based index
        src_col, dst_col (str): column names. Defaults are 'node_1' and 'node_2.
        weight_col (optional str): set this if weight matrix in needed.

    Returns:
        adjacency/weight matrix (csr_matrix)
    """
    row_indices = [src_index[u] for u in edges[src_col]]
    col_indices = [dst_index[v] for v in edges[dst_col]]
    if weight_col is None:
        weights = [1] * len(edges)
        dtype = np.int8
    else:
        weights = edges[weight_col]
        dtype = weights.dtype
    
    res = csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(len(src_index), len(dst_index)),
        dtype=dtype
    )
    return res


def extract_clustered_table(res, data):
    """
    input
    =====
    res:     <sns.matrix.ClusterGrid>  the clustermap object
    data:    <pd.DataFrame>            input table
    
    output
    ======
    returns: <pd.DataFrame>            reordered input table
    """
    
    # if sns.clustermap is run with row_cluster=False:
    if res.dendrogram_row is None:
        print("Apparently, rows were not clustered.")
        return -1
    
    if res.dendrogram_col is not None:
        # reordering index and columns
        new_cols = data.columns[res.dendrogram_col.reordered_ind]
        new_ind = data.index[res.dendrogram_row.reordered_ind]
        
        return data.loc[new_ind, new_cols]
    
    else:
        # reordering the index
        new_ind = data.index[res.dendrogram_row.reordered_ind]

        return data.loc[new_ind,:]
