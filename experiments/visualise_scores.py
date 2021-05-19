from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from utils import describe_mean_std, layer_density


METHOD_SELECTION = ['Random', 'MaxEnt', 'IPF', 'IPF enforced', 'DBCM', 'DBCM enforced']
METHOD_RENAMING = {
    'DBCM': 'f-DBCM',
    'DBCM enforced': 'f-DBCM *',
    'IPF enforced': 'IPF *'
}
PLOT_METHODS = ('MaxEnt', 'IPF enforced', 'DBCM enforced')
LATEX_COLUMNS = {
    'f1': r'$F_1\text{-score}$',
    'precision': 'Precision',
    'recall': 'Recall',
    'corr_in': r'$r_\text{in}$',
    'corr_out': r'$r_\text{out}$',
    'pv_in': r'$p\text{-value}_\text{in}$',
    'pv_out': r'$p\text{-value}_\text{out}$',
    's_in_js': r'$JSD_\text{in}$',
    's_out_js': r'$JSD_\text{out}$',
    'delta_L': r'$\delta_L$'
}
METRICS_CAPTION = 'Binary classification metrics for layer-wise reconstruction of {}'
CONSTRAINTS_CAPTION = \
    'Mean values of constraint compliance metrics for layerwise reconstruction of {}'
CONSTRAINT_COLS = ['corr_in', 'corr_out', 'pv_in', 'pv_out', 's_in_js', 's_out_js']

def plot_reconstruction_results(result_output: pd.DataFrame,
                                edges: pd.DataFrame,
                                methods: Iterable[str] = PLOT_METHODS):
    densities = edges.groupby('layer_id').apply(layer_density).rename('density')
    stats = result_output\
        .reset_index()\
        .groupby(['name', 'layer_id'])\
        .agg('mean')\
        .reset_index()
    stats = pd.merge(stats[stats.name.isin(methods)],
                     densities,
                     left_on='layer_id',
                     right_index=True)\
        .rename(columns={'f1': 'F1 score',
                         'density': 'Layer density',
                         'name': 'Method'})
    stats['Method'] = stats['Method'].replace(METHOD_RENAMING)
    sns.jointplot(data=stats,
                  x='Layer density',
                  y='F1 score',
                  hue='Method',
                  ylim=(-.05, 1.05 * min(1, result_output.f1.max())),
                  alpha=0.7)


def plot_error_for_L(result_output: pd.DataFrame):
    stat = result_output\
        .groupby('name')\
        .apply(lambda g:  _relative_error(g.num_expected.values,
                                          g.num_predicted.values))
    col_renaming = {'name': 'method', 0: 'relative error'}
    stat_df = stat.loc[METHOD_SELECTION]\
        .rename(index=METHOD_RENAMING)\
        .reset_index()\
        .rename(columns=col_renaming)\
        .explode('relative error')

    display(stat\
        .apply(describe_mean_std)\
        .to_frame()\
        .rename(columns=col_renaming))

    sns.set_theme(style="whitegrid")
    sns.boxplot(data=stat_df, x='method', y='relative error')


def generate_latex_summary(result_output: pd.DataFrame,
                           label: str,
                           dataset_name: str):
    metrics = result_output\
        [result_output.name.isin(METHOD_SELECTION)]\
        .dropna()\
        .groupby('name')

    rel_L_error = metrics\
        .apply(lambda g:  _mean_rel_error(g.num_expected.values,
                                          g.num_predicted.values))\
        .rename('delta_L')

    metrics[['f1', 'precision', 'recall']]\
        .agg(describe_mean_std(num_fmt=' {:.2f} '))\
        .reindex(METHOD_SELECTION)\
        .rename(index=METHOD_RENAMING, columns=LATEX_COLUMNS)\
        .to_latex(f'output/scores_{label}.tex',
                  index_names=False,
                  label=f'table:metrics_{label}',
                  bold_rows=True,
                  escape=False,
                  caption=METRICS_CAPTION.format(dataset_name))
        
    metrics[CONSTRAINT_COLS]\
        .agg('mean')\
        .join(rel_L_error)\
        .reindex(index=METHOD_SELECTION, columns=['delta_L'] + CONSTRAINT_COLS)\
        .rename(index=METHOD_RENAMING, columns=LATEX_COLUMNS)\
        .to_latex(f'output/constraints_{label}.tex',
                  index_names=False,
                  label=f'table:constraints_{label}',
                  bold_rows=True,
                  float_format='%.2f',
                  escape=False,
                  caption=CONSTRAINTS_CAPTION.format(dataset_name))


def _relative_error(t, p):
    return (p - t) / t


def _mean_rel_error(t, p):
    return np.mean(_relative_error(t, p))


if __name__ == '__main__':
    print('FAO reconstruction visualisation')
    
    from fao_data import load_all_layers
    
    eval_res = pd.read_csv('output/eval_all.csv')
    edges = load_all_layers()
    plot_reconstruction_results(eval_res, edges)
    plt.savefig('output/fao_f1.png', dpi=600)
    plt.savefig('output/fao_f1.svg')
    plt.close()
    
    plot_error_for_L(eval_res)
    plt.savefig('output/fao_L_err.png', dpi=600)
    plt.savefig('output/fao_L_err.svg')
    
    generate_latex_summary(eval_res, label='fao', dataset_name='the FAO network')
    
    print('Saved plots and tables in `output` folder')
