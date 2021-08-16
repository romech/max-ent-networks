import argparse
import logging
import os
import random
from contextlib import redirect_stdout
from datetime import datetime
from itertools import combinations
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map

from experiments.metrics import evaluate_reconstruction
from experiments.visualise_adjmatrix import adjmatrix_figure
from fao_data import load_dataset
from reconstruction import ipf, ipf_multiplex, random_baseline
from multiplex.correlation_metrics import count_common_links, multiplexity, multiplexity_eval_metrics
from multiplex.multilayer_sampling import MultiLayerSplit, multilayer_sample
from utils import (adjmatrix_to_df, adjmatrix_to_edgelist, display, edges_to_matrix, fallback, filter_by_layer,
                   index_elements, matrix_intersetions, describe_mean_std)


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename='output/log/fao_multireconstruct.log',
                    filemode='a')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


METRICS_DISPLAY = ['f1', 'precision', 'recall', 'L_mape',
                #    'mltplx_mae', 'mltplx_mape',
                #    'mltrcp_mae', 'mltrcp_mape',
                #    'corr_in', 'corr_out',
                #    'pv_in', 'pv_out',
                #    's_in_js', 's_out_js'
                   's_in_mae', 's_out_mae',
                   's_in_mape', 's_out_mape',
                #    'r2_in', 'r2_out',
                #    'spcorr_in', 'spcorr_out',
                #    'sp_pv_in', 'sp_pv_out',
                   ]

def fao_multilayer_sample(layer_ids: Optional[List[int]] = None,
                          hidden_ratio: float = 0.5,
                          random_state: Optional[int] = None) -> MultiLayerSplit:
    dataset = load_dataset(drop_small_layers=True)
    if layer_ids is None:
        layer_ids = random.sample(dataset.layer_names.index.tolist(), k=2)
    elif len(layer_ids) != 2:
        raise NotImplementedError('Only two layers at once supported yet')
    
    edges = filter_by_layer(dataset.edges, layer_ids)
    
    return multilayer_sample(
        edges=edges,
        layer_ids=layer_ids,
        hidden_ratio=hidden_ratio,
        random_state=random_state
    )


def demo_single_run(layer_ids=None):    
    sample = fao_multilayer_sample(layer_ids=layer_ids)
    layer_ids = sample.layer_ids
    sample.print_summary()
    pred_matrices, ipf_outputs = ipf_multiplex.two_layer_ipf(
        sample, ipf_steps=10, return_ipf_outputs=True)
    
    print('1 - Evaluating layers individually')
    print('The results are expected to contain no significant difference '
            'between baseline and tuned variations')
    metrics_df = []
    for layer_id, ipf_matrix, tuned_matrix in zip(layer_ids, ipf_outputs, pred_matrices):
        layer_sample = sample.select_single(layer_id, all_nodes=True)
        ipf_metrics = evaluate_reconstruction(layer_sample, ipf_matrix)
        tuned_metrics = evaluate_reconstruction(layer_sample, tuned_matrix)
        metrics_df.append(pd.DataFrame({
            f'Baseline (layer {layer_id})': ipf_metrics,
            f'Tuned (layer {layer_id})': tuned_metrics
        }).T)
        
        display(metrics_df[-1][METRICS_DISPLAY])
    
    print('2 - Evaluating multiplexity tuning')
    print('There should be a significant decrease in error values')
    ipf_pred_edges = [adjmatrix_to_df(mat, sample.node_index, layer_id)
                      for layer_id, mat in zip(layer_ids, ipf_outputs)]
    tuned_pred_edges = [adjmatrix_to_df(mat, sample.node_index, layer_id)
                        for layer_id, mat in zip(layer_ids, pred_matrices)]
    ipf_pred_edges = pd.concat(ipf_pred_edges, ignore_index=True)
    tuned_pred_edges = pd.concat(tuned_pred_edges, ignore_index=True)
    
    mlt_metrics = pd.DataFrame({
        '(i)': multiplexity_eval_metrics(sample, ipf_pred_edges),
        '(ii)': multiplexity_eval_metrics(sample, tuned_pred_edges)}).T
    print(mlt_metrics)
    
    common_links = count_common_links(sample.full.edges, *sample.layer_ids)
    print('Target common links:', common_links)
    print('Target multiplexity score',  mlt_metrics['mltplx_tgt'].iloc[0])
    
    # Metrics to save for further analysis
    eval_report = {
        'layer a': layer_ids[0],
        'layer b': layer_ids[1],
        'total nodes': len(sample.node_index),
        'target common links': common_links,
        'target multiplexity': mlt_metrics.iloc[0].mltplx_tgt,
        'baseline multiplexity': mlt_metrics.iloc[0].mltplx_pred,
        'baseline mltplx mae': mlt_metrics.iloc[0].mltplx_mae,
        'baseline mltplx mape': mlt_metrics.iloc[0].mltplx_mape,
        'tuned multiplexity': mlt_metrics.iloc[1].mltplx_pred,
        'tuned mltplx mae': mlt_metrics.iloc[1].mltplx_mae,
        'tuned mltplx mape': mlt_metrics.iloc[1].mltplx_mape,
        'f1 increase': (metrics_df[0].f1.pct_change().iloc[1] +
                        metrics_df[1].f1.pct_change().iloc[1]) / 2,
    }
    return eval_report


def demo_single_run_silent(layer_ids=None):
    log_path = 'output/log/{}_{}.log'.format(datetime.now().date(), os.getpid())
    with open(log_path, 'a') as log_file, redirect_stdout(log_file):
        try:
            return demo_single_run(layer_ids)
        except ValueError as e:
            if str(e) != 'probabilities do not sum to 1':
                raise e
            else:
                return None


def evaluate_many(n='all', workers=0):
    reports = []
    output_file = f'output/fao_report_{n}.csv'
    if n == 'all':
        dataset = load_dataset(drop_small_layers=True)
        layer_pairs = list(combinations(dataset.layer_names.index, r=2))
    else:
        layer_pairs = [None] * n
        
    if workers == 0:
        for layer_pair in tqdm(layer_pairs):
            try:
                reports.append(demo_single_run_silent(layer_pair))
            except KeyboardInterrupt:
                output_file = output_file.replace('.csv', '_partial.csv')
                break
    else:
        reports = process_map(demo_single_run_silent,
                              layer_pairs,
                              max_workers=workers,
                              chunksize=64)
        
    reports = list(filter(None, reports))
    print('Total completed', len(reports), 'out of', len(layer_pairs))
            
    df = pd.DataFrame.from_records(reports)
    df.to_csv(output_file, index=False)
    
    analysis = {
        'Improvement in multiplexity MAE':
            np.quantile(df['baseline mltplx mae'].values - df['tuned mltplx mae'].values, 0.5),
        'Improvement in multiplexity MAPE':
            np.quantile(df['baseline mltplx mape'].values - df['tuned mltplx mape'].values, 0.5),
        'Improvement in F1 (relative)': df['f1 increase'].mean()
    }
    print(pd.Series(analysis))
    return df

    
if __name__ == '__main__':
    pd.set_option('precision', 2)
    
    # evaluate_many(10, workers=2)
    evaluate_many(workers=4)
    
    # print(demo_single_run())
    # print(demo_single_run([119, 279]))
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-l', '--layer_id', type=int, default=None,
    #                     help='Run single experiment with specified layer_id')
    # parser.add_argument('-a', '--all', action='store_true',
    #                     help='Run experiments for all layers')
    # parser.add_argument('-n', type=int, default=30,
    #                     help='Run experiments for n layers')
    # parser.add_argument('-ll', '--layer_ids', type=int, nargs='*',
    #                     help='Run for selected layer ids')
    # parser.add_argument('-s', '--num_seeds', type=int, default=3,
    #                     help='Number of random seeds')
    # parser.add_argument('-w', '--num_workers', type=int, default=6)

    # args = parser.parse_args()
    
    # if args.layer_id is not None:
    #     demo_random_single_layer(args.layer_id)
    # elif args.all:
    #     res = demo_evaluate_multiple_layers(num_seeds=args.num_seeds,
    #                                         num_workers=args.num_workers)
    #     res.to_csv('output/eval_all.csv')
    # elif args.layer_ids:
    #     res = demo_evaluate_multiple_layers(layer_ids=args.layer_ids,
    #                                         num_seeds=args.num_seeds,
    #                                         num_workers=args.num_workers)
    #     res.to_csv('output/eval.csv')
    # else:
    #     res = demo_evaluate_multiple_layers(n=args.n,
    #                                         num_seeds=args.num_seeds,
    #                                         num_workers=args.num_workers)
    #     res.to_csv('output/eval.csv')
