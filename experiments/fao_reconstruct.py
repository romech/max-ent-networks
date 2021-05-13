import argparse
import logging
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from experiments.metrics import evaluate_reconstruction
from experiments.visualise_adjmatrix import adjmatrix_figure
from fao_data import load_dataset
from reconstruction import dbcm, ipf, random_baseline
from sampling import LayerSplit, layer_split_with_no_observables, random_layer_split
from utils import (display, edges_to_matrix, filter_by_layer,
                   index_elements, matrix_intersetions, describe_mean_std)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


METRICS_DISPLAY = ['f1', 'precision', 'recall',
                   'corr_in', 'corr_out',
                   'p_val_in', 'p_val_out',
                   's_in_js', 's_out_js']

def fao_layer_sample(layer_id=None, hidden_ratio=0.5, random_state=None) -> LayerSplit:
    """
    Select random layer, then split its nodes into 'observed' and 'hidden' parts.
    Edges that are adjacent to any hidden node are also considered as 'hidden'.

    Args:
        layer_id (int, optional): Chosen at random by default.
        hidden_ratio (float): Ratio of nodes hidden (0.25 by default).
                              Ratio of hidden edges is essentially larger.
                              If ratio = 1.0 then the observed edge list is empty, and
                              reconstruction algorithm does not account for them.
        random_state (int): Random seed for splitting nodes.

    Returns:
        LayerSplit: data class containing nodes and edges for each partition:
                    'observed', 'hidden' and 'full'.
    """
    dataset = load_dataset()
    if layer_id is None:
        layer_id = random.choice(dataset.layer_names.index)
    
    edges = filter_by_layer(dataset.edges, layer_id)
    
    if hidden_ratio != 1:
        return random_layer_split(
            edges=edges,
            layer_id=layer_id,
            hidden_ratio=hidden_ratio,
            random_state=random_state
        )
    else:
        return layer_split_with_no_observables(
            edges=edges,
            layer_id=layer_id
        )


def demo_evaluate_multiple_layers(n=None, layer_ids=None, num_seeds=2, num_workers=6):
    if layer_ids is None:
        layer_ids = load_dataset(drop_small_layers=True).layer_names.index
        if n is not None:
            layer_ids = random.choices(layer_ids, k=n)
    
    seeds = np.arange(num_seeds)
    experiments = [
        ('Random', random_baseline.reconstruct_layer_sample),
        ('MaxEnt', ipf.reconstruct_layer_sample(ipf_steps=0)),
        ('IPF', ipf.reconstruct_layer_sample_unconsciously),
        ('IPF enforced', ipf.reconstruct_layer_sample),
        ('IPF v2', ipf.reconstruct_v2),
        ('DBCM', dbcm.reconstruct_layer_sample(enforce_observed=False)),
        ('DBCM enforced', dbcm.reconstruct_layer_sample(enforce_observed=True)),
    ]
    
    index_keys = []
    runs = []
    for layer_id in layer_ids:
        for seed in seeds:
            sample = fao_layer_sample(layer_id, random_state=seed)
            for name, reconstruct_func in experiments:
                index_keys.append((layer_id, name, seed))
                runs.append((sample, reconstruct_func))
    
    results_list = process_map(_run_single_eval, runs,
                               chunksize=3, max_workers=num_workers, smoothing=0)
    results_df = pd.DataFrame(
        results_list,
        index=pd.MultiIndex.from_tuples(index_keys, names=['layer_id', 'name', 'seed']),
    )
    print('Stats by layer')
    display(results_df[METRICS_DISPLAY].groupby(level=['layer_id', 'name']).agg(describe_mean_std))
    
    print('Stats by method')
    display(results_df[METRICS_DISPLAY].groupby(level=['name']).agg(describe_mean_std))
    return results_df


def _run_single_eval(params):
    sample, reconstruct_func = params
    p_ij = reconstruct_func(sample)
    res = evaluate_reconstruction(sample, p_ij)
    return res


def demo_random_single_layer(layer_id=None):
    # tiny: layer_id=288
    # small: layer_id=202
    sample = fao_layer_sample(layer_id=layer_id)
    sample.print_summary()
    
    predictions = OrderedDict()
    eval_res = OrderedDict()
    experiments = [
        ('Random', random_baseline.reconstruct_layer_sample),
        ('MaxEnt', ipf.reconstruct_layer_sample(ipf_steps=0)),
        ('IPF', ipf.reconstruct_layer_sample_unconsciously),
        # ('IPF enforced', ipf.reconstruct_layer_sample),
        ('IPF enforced', ipf.reconstruct_v2),
        ('DBCM', dbcm.reconstruct_layer_sample(enforce_observed=False)),
        ('DBCM enforced', dbcm.reconstruct_layer_sample(enforce_observed=True)),
    ]
    
    with tqdm(experiments, desc='Reconstruction experiments') as experiments_pbar:
        for name, reconstruction_func in experiments_pbar:
            experiments_pbar.set_postfix_str(name)
            p_ij = reconstruction_func(sample)
            eval_res[name] = evaluate_reconstruction(sample, p_ij)
            predictions[name] = p_ij
    
    eval_res = pd.DataFrame.from_dict(eval_res, orient='index')
    print(eval_res)
    
    # Visualisation
    node_index = index_elements(sample.full.nodes)
    target_w = edges_to_matrix(sample.full.edges, node_index, node_index)
    demo_sample = np.empty(target_w.shape, dtype=np.int8)
    target_w.todense(out=demo_sample)
    
    observed_entries = matrix_intersetions(sample.observed.nodes, sample.node_index)
    observed_mask = np.ones_like(demo_sample)
    observed_mask[observed_entries] = 2
    demo_sample = (demo_sample * observed_mask).astype(np.float32) / 2
    
    res = [
        ('Source\n(observed - yellow,\nhidden - pink)', demo_sample),
        ('IPF', predictions['IPF']),
        ('IPF\n(links enforced)', predictions['IPF enforced']),
        # ('Random', random_baseline.reconstruct_layer_sample),
        ('MaxEnt', predictions['MaxEnt']),
        ('DBCM', predictions['DBCM']),
        ('DBCM\n(links enforced)', predictions['DBCM enforced']),
    ]
    adjmatrix_figure(
        res,
        title=f'Reconstruction results – probability matrices (layer {sample.layer_id})',
    )
    plt.savefig('output/fao_reconst_comparison.png', dpi=1200)
    plt.savefig('output/fao_reconst_comparison.svg')
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pd.set_option('precision', 2)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--layer_id', type=int, default=None,
                        help='Run single experiment with specified layer_id')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Run experiments for all layers')
    parser.add_argument('-n', type=int, default=30,
                        help='Run experiments for n layers')
    parser.add_argument('-ll', '--layer_ids', type=int, nargs='*',
                        help='Run for selected layer ids')
    parser.add_argument('-s', '--num_seeds', type=int, default=2,
                        help='Number of random seeds')
    parser.add_argument('-w', '--num_workers', type=int, default=6)

    args = parser.parse_args()
    
    if args.layer_id is not None:
        demo_random_single_layer(args.layer_id)
    elif args.all:
        res = demo_evaluate_multiple_layers(num_seeds=args.num_seeds,
                                            num_workers=args.num_workers)
        res.to_csv('output/eval_all.csv')
    elif args.layer_ids:
        res = demo_evaluate_multiple_layers(layer_ids=args.layer_ids,
                                            num_seeds=args.num_seeds,
                                            num_workers=args.num_workers)
        res.to_csv('output/eval.csv')
    else:
        res = demo_evaluate_multiple_layers(n=args.n,
                                            num_seeds=args.num_seeds,
                                            num_workers=args.num_workers)
        res.to_csv('output/eval.csv')
