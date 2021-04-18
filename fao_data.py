from pathlib import Path
from typing import Iterable, Union, Tuple

import pandas as pd

from utils import replace_underscores

DATASET_PATH = Path('FAO_dataset/Dataset')
EDGES_PATH = DATASET_PATH / 'fao_trade_multiplex.edges'
NODES_PATH = DATASET_PATH / 'fao_trade_nodes.txt'
LAYERS_PATH = DATASET_PATH / 'fao_trade_layers.txt'


def load_all_layers() -> pd.DataFrame:
    df = pd.read_csv(
        EDGES_PATH,
        names=('layer_id', 'node_1', 'node_2', 'weight'),
        sep=' '
    )
    return df


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    edges = load_all_layers()
    node_names = pd.read_csv(NODES_PATH, sep=' ', index_col='nodeID')
    node_names = node_names['nodeLabel'].apply(replace_underscores)
    layer_names = pd.read_csv(LAYERS_PATH, sep=' ', index_col='layerID')
    layer_names = layer_names['layerLabel'].apply(replace_underscores)
    return edges, node_names, layer_names


def load_layers(layer_id: Union[int, Iterable[int]]) -> pd.DataFrame:
    iter_csv = pd.read_csv(
        EDGES_PATH,
        names=('layer_id', 'node_1', 'node_2', 'weight'),
        sep=' ',
        iterator=True,
        chunksize=1000
    )
    if not isinstance(layer_id, (tuple, list)):
        layer_id = [layer_id]
    df = pd.concat([chunk[chunk['layer_id'].isin(layer_id)] for chunk in iter_csv])
    # df = df.drop(columns='layer_id')
    return df
