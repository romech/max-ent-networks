import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import layer_density


def plot_reconstruction_results(result_output: pd.DataFrame,
                                edges: pd.DataFrame,
                                methods=('Random', 'IPF', 'DBCM')):
    densities = edges.groupby('layer_id').apply(layer_density).rename('density')
    stats = result_output\
        .groupby(level=['name', 'layer_id'])\
        .agg('mean')\
        .reset_index()
    stats = pd.merge(stats[stats.name.isin(methods)],
                     densities,
                     left_on='layer_id',
                     right_index=True)\
        .rename(columns={'f1': 'F1 score',
                         'density': 'Layer density',
                         'name': 'Method'})
    sns.jointplot(data=stats,
                  x='Layer density',
                  y='F1 score',
                  hue='Method',
                  ylim=(-.05, 1.05 * min(1, result_output.f1.max())))


def fao_plot_reconstruction():
    from fao_data import load_all_layers
    from experiments.fao_reconstruct import demo_evaluate_multiple_layers
    
    edges = load_all_layers()
    result_output = demo_evaluate_multiple_layers(num_seeds=5)
    plot_reconstruction_results(result_output, edges)
    plt.savefig('output/fao_f1_layers.svg')


if __name__ == '__main__':
    fao_plot_reconstruction()
