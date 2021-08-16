from itertools import combinations, repeat
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import fao_data
from multiplex.correlation_metrics import multiplexity


REPORT_CSV_FAO_OLD = 'output/mltplx_report.csv'
REPORT_CSV_FAO = 'output/fao_report_all.csv'
memory = Memory('.cache/', verbose=2)



def flatten_df_for_seaborn(df: pd.DataFrame,
                           columns: Union[Dict[str, str], List[str]],
                           xlabel='experiment', ylabel='value'):
    """
    Transform dataframe with columns A, B, C into dataframe like
    ```
    ylabel | xlabel
    ---------------
      a_1  |   "A"
      a_2  |   "A"
     ..... |  .....
      c_n  |   "C"
    ```

    Args:
        df : dataframe
        columns (list or dict): columns to use
        xlabel : new column name for labels.
        ylabel : new column name for values.

    Returns:
        flat_df : dataframe that can be plotted with seaborn like
            `sns.boxplot(data=flat_df, x=xlabel, y=ylabel)`
    """    

    sub_dfs = []
    if not isinstance(columns, dict):
        columns = {c: c for c in columns}
    for col_name, display_name in columns.items():
        col = df[col_name].rename(ylabel).to_frame()
        col[xlabel] = display_name
        sub_dfs.append(col)
    return pd.concat(sub_dfs, ignore_index=True)


def multiplexity_error_plots(report_csv):
    df = pd.read_csv(report_csv)
    df = drop_extreme_outliers(
        df, ['baseline mltplx mae', 'baseline mltplx mape',
             'tuned mltplx mae', 'tuned mltplx mape'])
    mae_df = flatten_df_for_seaborn(df, {
        'baseline mltplx mae': 'Baseline approach',
        'tuned mltplx mae': 'Multiplexity-tuned'
    })
    mape_df = flatten_df_for_seaborn(df, {
        'baseline mltplx mape': 'Baseline approach',
        'tuned mltplx mape': 'Multiplexity-tuned'
    })
    sns.set_theme(style="ticks")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ########
    sns.boxplot(x='experiment', y='value', data=mae_df,
                whis=2, width=.6, palette="vlag", fliersize=3, ax=ax1)

    ax1.yaxis.grid(True)
    ax1.set(ylabel="absolute error", xlabel="")
    ax1.set_title('Absolute error of multiplexity value')
    ########
    # ax2.set_yscale("log")
    sns.boxplot(x='experiment', y='value', data=mape_df,
                whis=2, width=.6, palette="vlag", fliersize=3, ax=ax2)

    ax2.yaxis.grid(True)
    ax2.set(ylabel="relative error", xlabel="")
    ax2.set_title('Relative error of multiplexity value')
    ########
    fig.tight_layout(pad=2.)
    plt.savefig('output/mltplx.svg')
    plt.show()


def drop_extreme_outliers(df, cols, top=0.99):
    for col_name in cols:
        max_val = df[col_name].quantile(top)
        df = df[df[col_name] <= max_val]
    return df


def multiplexity_func(args):
    edges, layer_ids = args
    return multiplexity(edges, *layer_ids)


def compute_multiplexity(dataset) -> List[float]:
    layer_pairs = list(zip(
        repeat(dataset.edges),
        combinations(dataset.layer_names.index, r=2)
    ))
    multiplexities = process_map(multiplexity_func,
                                 layer_pairs,
                                 chunksize=256,
                                 max_workers=4)
    return multiplexities


def plot_multiplexity(dataset=None):
    if dataset is None:
        dataset = fao_data.load_dataset(drop_small_layers=True)
    n = len(dataset.layer_names)
    multiplexities = compute_multiplexity(dataset)
    multiplexities_2d = squareform(multiplexities) + np.eye(n)
    plt.figure(figsize=(4, 3.4), dpi=200)
    plt.imshow(multiplexities_2d, cmap='Greens')
    plt.colorbar(shrink=0.95)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig('output/mltplx_heatmap.svg')
    plt.savefig('output/mltplx_heatmap.png', dpi=1200)
    plt.show()


def illustrations_for_tuning_algo():
    n = 14
    savedir = 'deliverables/algo_illustration/'
    pa = np.random.rand(n, n) ** 4
    pb = np.random.rand(n, n) ** 4
    pj = pa * pb
    aj = np.random.rand(n, n) < pj
    
    for m, name in [(pa, 'p_a'), (pb, 'p_b'), (pj, 'p_ab'), (aj, 'a_ab')]:
        plt.figure(figsize=(3,3))
        plt.axis('off')
        plt.imshow(m, cmap='Greens', vmin=0, vmax=1)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
        plt.savefig(savedir + name + '.svg', pad_inches=0)
    
    

def plot_multiplexity_histogram(dataset=None):
    if dataset is None:
        dataset = fao_data.load_dataset(drop_small_layers=True)
    multiplexities = compute_multiplexity(dataset)
    plt.rc('font', size=14)
    plt.hist(multiplexities, bins=32, facecolor='g', alpha=0.75)
    plt.ylabel('frequency')
    plt.xlabel('multiplexity value')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticks([])
    plt.tight_layout()
    plt.savefig('output/mltplx_hist.svg')
    plt.show()


def plot_mltplx_gain(report_csv):
    df = pd.read_csv(report_csv)
    gain = df['baseline mltplx mae'] - df['tuned mltplx mae']
    
    sns.set_style('whitegrid')
    plt.rc('font', size=12)
    
    ax = sns.jointplot(x=df['total nodes'], y=gain,
                  marker='+', color='g', kind='reg', order=2,
                  scatter_kws=dict(alpha=0.3),
                  line_kws=dict(color='darkgreen'))
    ax.ax_joint.set(ylabel="absolute error decrease", xlabel="number of nodes present on layers")
    ax.fig.suptitle('Effect of multiplexity tuning approach')
    ax.fig.tight_layout()
    plt.savefig('output/mltplx_gain.svg')
    plt.show()


def compute_metrics(report_csv):
    from scipy.stats import wilcoxon
    
    df = pd.read_csv(report_csv)
    df = drop_extreme_outliers(df, ['baseline mltplx mape', 'tuned mltplx mape'])
    print(df[['baseline mltplx mae', 'baseline mltplx mape',
              'tuned mltplx mae', 'tuned mltplx mape', 'f1 increase']].describe())
    df['baseline mltplx mae'].mean()
    analysis = {
        'Improvement in multiplexity MAE':
            np.quantile(df['baseline mltplx mae'].values - df['tuned mltplx mae'].values, 0.5),
        'Improvement in multiplexity MAPE':
            np.quantile(df['baseline mltplx mape'].values - df['tuned mltplx mape'].values, 0.5),
        'Improvement in F1 (relative)': df['f1 increase'].mean(),
        'Average error (i)': (df['baseline multiplexity'] - df['target multiplexity']).mean(),
        'Average error (ii)': (df['tuned multiplexity'] - df['target multiplexity']).mean(),
    }
    print(pd.Series(analysis))
    
    print('H0: mean = 0', wilcoxon(df['f1 increase'])[1])
    print('H0: mean < 0', wilcoxon(df['f1 increase'], alternative='greater')[1])

if __name__ == '__main__':
    compute_multiplexity = memory.cache(compute_multiplexity)
    
    # 1. Box plots showing that multiplexity tuning approach yields
    # more truthful multiplexity scores
    
    # multiplexity_error_plots(REPORT_CSV_FAO)
    
    # 2.1. Multiplexity heatmap
    # plot_multiplexity()
    
    # 2.2. Multiplexity histogram
    # plot_multiplexity_histogram()
    
    # 3. Plot multiplexity tuning approach gains
    # plot_mltplx_gain(REPORT_CSV_FAO)
    
    # 4. Metrics
    # compute_metrics(REPORT_CSV_FAO)
    
    
    # illustrations_for_tuning_algo()