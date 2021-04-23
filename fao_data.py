import logging
from functools import lru_cache as cached
from pathlib import Path
from typing import Iterable, NamedTuple, Union

import pandas as pd

from utils import replace_underscores, filter_by_layer

DATASET_PATH = Path('FAO_dataset/Dataset')
EDGES_PATH = DATASET_PATH / 'fao_trade_multiplex.edges'
NODES_PATH = DATASET_PATH / 'fao_trade_nodes.txt'
LAYERS_PATH = DATASET_PATH / 'fao_trade_layers.txt'


class FaoDataset(NamedTuple):
    edges: pd.DataFrame    # columns: 'layer_id', 'node_1', 'node_2', 'weight'
    node_names: pd.Series  # column: 'nodeLabel'
    layer_names: pd.Series # column: 'layerLabel'


def load_all_layers() -> pd.DataFrame:
    df = pd.read_csv(
        EDGES_PATH,
        names=('layer_id', 'node_1', 'node_2', 'weight'),
        sep=' '
    )
    return df


@cached
def load_dataset(drop_small_layers=False) -> FaoDataset:
    """
    Load FAO dataset including edges, layer_names and layer names.

    Args:
        drop_small_layers (bool): whether to drop layer with less than 50 edges. 

    Returns:
        FaoDataset: tuple containing (edges, node_names, layer_names)
    """
    edges = load_all_layers()
    node_names = pd.read_csv(NODES_PATH, sep=' ', index_col='nodeID')
    node_names = node_names['nodeLabel'].apply(replace_underscores)
    layer_names = pd.read_csv(LAYERS_PATH, sep=' ', index_col='layerID')
    layer_names = layer_names['layerLabel'].apply(replace_underscores)
    
    if drop_small_layers:
        edges, layer_names = _drop_small_layers(edges, layer_names)

    return FaoDataset(edges, node_names, layer_names)


def load_layers(layer_id: Union[int, Iterable[int]]) -> pd.DataFrame:
    """
    Faster way to load subset of layers

    Args:
        layer_id (int or list[int]): one or more layer ids

    Returns:
        edges (pd.DataFrame)
    """
    iter_csv = pd.read_csv(
        EDGES_PATH,
        names=('layer_id', 'node_1', 'node_2', 'weight'),
        sep=' ',
        iterator=True,
        chunksize=1000
    )
    df = pd.concat([filter_by_layer(chunk, layer_id) for chunk in iter_csv])
    # df = df.drop(columns='layer_id')
    return df


def _drop_small_layers(edges, layer_names, k=50):
    """
    Drop layers which contain less than k edges.
    Remove from both `edges` and `layer_names`.

    Returns:
        edges, layer_names
    """
    total_layers = len(layer_names)
    edge_count = edges.layer_id.value_counts(sort=False).rename('num_edges')
    edge_count = layer_names.to_frame().join(edge_count, how='left')
    is_small_layer = edge_count.num_edges < k
    layer_names = layer_names[~is_small_layer]
    edges = filter_by_layer(edges, layer_names.index)
    logging.debug('Loaded %d layers, %d dropped' %
                  (total_layers, total_layers - len(layer_names)))
    return edges, layer_names


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    load_dataset(drop_small_layers=True)
