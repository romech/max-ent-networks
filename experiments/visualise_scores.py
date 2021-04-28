import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import layer_density


def plot_reconstruction_results(result_output: pd.DataFrame,
                                edges: pd.DataFrame,
                                methods=('Random', 'IPF', 'DBCM')):
    densities = edges.groupby('layer_id').apply(layer_density)
    stats = result_output[result_output.name.isin(methods)]\
        .groupby(level=['name', 'layer_id'])\
        .agg('mean')\
        .reset_index()
    stats = pd.merge(stats, densities, left_on='layer_id', right_index=True)\
        .rename(columns={'f1': 'F1 score',
                         'density': 'Layer density',
                         'name': 'Method'})
    sns.jointplot(data=stats, x='Layer density', y='F1 score', hue='Method')
    plt.savefig('output/f1_layers.svg')


def fao_plot_reconstruction():
    from fao_data import load_all_layers
    from experiments.fao_reconstruct import demo_evaluate_multiple_layers
    
    edges = load_all_layers()
    result_output = demo_evaluate_multiple_layers(num_seeds=5)
    plot_reconstruction_results(result_output, edges)


if __name__ == '__main__':
    fao_plot_reconstruction()
