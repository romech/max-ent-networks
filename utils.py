from itertools import chain
import traceback
from typing import Set

import pandas as pd
import toolz


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
