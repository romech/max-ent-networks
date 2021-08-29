import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory
from sklearn.cluster import OPTICS, AgglomerativeClustering
from tqdm import tqdm

from fao_data import load_all_layers, load_dataset
from fao_analysis.cluster_analysis import demo_elbow_method, try_clustering
from fao_analysis.inter_layer import pairwise_multiplexity
from utils import (display, node_set_size, put_col_in_front,
                   replace_underscores, extract_clustered_table)

memory = Memory('.cache', verbose=0)


def report():
    edges, node_names, layer_names = load_dataset()
    print('Total nodes: {}, edges: {}, layers: {}'.format(
        len(node_names), len(edges), len(layer_names)))
    layer_names.rename('product', inplace=True)
    node_names.rename('country', inplace=True)

    grouped = edges.groupby('layer_id')
    stats = grouped.agg(
        num_suppliers=('node_1', pd.Series.nunique),
        num_consumers=('node_2', pd.Series.nunique),
        num_edges=('weight', 'count')
    )
    stats['num_nodes'] = grouped.apply(node_set_size)
    
    for top_by in ('num_edges', 'num_suppliers', 'num_consumers'):
        top = stats.nlargest(10, top_by).join(layer_names)
        top = put_col_in_front(top, 'product').rename(columns=replace_underscores)
        print('LAYERS TOP BY', top_by.upper())
        display(top)
    
    for col, role in [('node_1', 'SUPPLY PRODUCTS'), ('node_2', 'CONSUME PRODUCTS')]:
        top = edges.groupby(col)\
            .agg(num_layers=('layer_id', pd.Series.nunique))\
            .nlargest(10, 'num_layers')\
            .join(node_names)
        print('NODES TOP BY', role)
        display(top)
    
    top = edges['node_1'].value_counts().iloc[:10].rename('degree').to_frame().join(node_names)
    print('TOP SUPPLIERS')
    display(top)
    
    top = edges['node_2'].value_counts().iloc[:10].rename('degree').to_frame().join(node_names)
    print('TOP CONSUMERS')
    display(top)


@memory.cache
def compute_fao_multiplexity(marginalized=True, log_weight=False):
    edges = load_all_layers()
    if log_weight:
        edges['weight'] = edges['weight'].apply(np.log)
        print('Setting marginalized=False')
        marginalized = False
    multiplexity_matrix, layer_ids = pairwise_multiplexity(edges, marginalized=marginalized)
    return multiplexity_matrix, layer_ids


def demo_multiplexity_histogram():
    multiplexity_matrix, _ = compute_fao_multiplexity()
    
    layer_pairs = np.tril_indices_from(multiplexity_matrix, -1)
    mult_pairs = multiplexity_matrix[layer_pairs]
    ax = sns.histplot(mult_pairs, element="step")
    ax.set(xlabel='Multiplexity scores')
    plt.show()


def demo_multiplexity(plot=True):
    _, _, layer_names = load_dataset()
    multiplexity_matrix, layer_ids = compute_fao_multiplexity()

    if plot:
        import scipy.spatial.distance as ssd
        from scipy.cluster import hierarchy
        
        labels = list(layer_names)
        dist_matrix = 1-multiplexity_matrix
        dist_array = ssd.squareform(dist_matrix)
        dist_linkage = hierarchy.linkage(dist_array)
        
        df = pd.DataFrame(dist_matrix, index=labels, columns=labels)
        cm = sns.clustermap(df,
                            row_linkage=dist_linkage,
                            col_linkage=dist_linkage,
                            figsize=(10, 10))
        plt.show()
        
        hmap = extract_clustered_table(cm, df).values
        scaled_hmap = np.repeat(np.repeat(hmap, 4, axis=0), 4, axis=1)
        mpl.image.imsave('plots/fao_cmap.svg', scaled_hmap)

    return multiplexity_matrix, layer_ids


def demo_clustering():
    _, _, layer_names = load_dataset()
    multiplexity_matrix, layer_ids = compute_fao_multiplexity()

    methods = {
        'OPTICS (min_samples=2)':
            OPTICS(min_samples=2, metric='precomputed'),
        'OPTICS (min_samples=3)':
            OPTICS(min_samples=3, metric='precomputed'),
        'Agglomerative clustering':
            AgglomerativeClustering(n_clusters=18, linkage='average', affinity='precomputed'),
    }
    labels = layer_names.loc[layer_ids]
    
    trials = {name: try_clustering(method, multiplexity_matrix, labels, verbose=False)
              for name, method in tqdm(methods.items(), desc='Clustering', unit='method')}
    
    table = pd.DataFrame({name: res.__dict__ for name, res in trials.items()}).T
    display(table[['num_clusters', 'num_outliers', 'score']].rename(columns=replace_underscores))
    return multiplexity_matrix, layer_ids, trials


def demo_clustering_coef(plot=True, num_workers=4):
    from fao_analysis.nx_toolset import clust_coef_by_layer
    
    label_clust = 'clustering coefficient'
    label_links = 'number of links'
    label_nodes = 'number of nodes'
    
    dataset = load_dataset()
    edges = dataset.edges
    clust_coef = clust_coef_by_layer(edges, num_workers=num_workers)
    clust_coef = pd.Series(clust_coef, name=label_clust)
    
    layer_grouped = edges.groupby('layer_id')
    links_num = layer_grouped\
        .agg(e=('weight', 'count'))\
        .rename(columns={'e': label_links})
    nodes_num = layer_grouped.apply(node_set_size).rename(label_nodes)
    
    df = dataset.layer_names.\
        to_frame().\
        join(clust_coef).\
        join(links_num).\
        join(nodes_num)
    display(df)
    
    if plot:
        g = sns.JointGrid(data=df, x=label_clust, y=label_links, space=0, ratio=14, height=7)
        g.ax_joint.set(yscale="log")
        g.plot_joint(sns.scatterplot, size=nodes_num, sizes=(10, 100),
                     color="g", alpha=.5, legend='auto')
        g.plot_marginals(sns.histplot, color="g", kde=True)
    
    return df
    

def demo_link_weights():
    edges = load_all_layers()
    ax = sns.histplot(edges, x='weight', bins=22, stat='probability', log_scale=True)
    ax.set(xlabel='link weights', ylabel='ratio')
    plt.show()


def demo_layers_embedding(interactive_mds=True):
    from sklearn.decomposition import TruncatedSVD
    
    if interactive_mds:
        import plotly.express as px
        from sklearn.manifold import MDS
    
    multiplexity_matrix, _ = compute_fao_multiplexity()
    dissim = multiplexity_matrix.max() - multiplexity_matrix
    n_layers = len(multiplexity_matrix)
    
    svd = TruncatedSVD(n_components=n_layers-1)
    svd.fit(dissim)
    expl_variance = np.cumsum(svd.explained_variance_ratio_)
    x_ticks = np.arange(1, n_layers)
    threshold = np.searchsorted(expl_variance, 0.95) + 1

    plt.title('SVD – Explained variance ratio')
    plt.plot(x_ticks, expl_variance)
    plt.axvline(x=threshold, label=f'95% explained variance,\n{threshold} components',
                linestyle='--', alpha=0.5, color='b')
    plt.axhline(y=0.95, linestyle='--', alpha=0.5, color='b')
    plt.xlabel('Reduced dimensionality')
    plt.legend()
    
    if not interactive_mds:
        return
    
    mds = MDS(
        n_components=2,
        dissimilarity='precomputed',
        n_init=100,
        n_jobs=6,
        max_iter=500,
        verbose=2
    )
    embed = mds.fit_transform(dissim)

    df = pd.DataFrame({'x': embed[:,0], 'y': embed[:,1]}, index=layer_names)
    fig = px.scatter(
        df, x='x', y='y',
        hover_name=layer_names,
        hover_data={'x': False, 'y': False},
        title='Multidimensional scailng'
    )
    fig.show()


if __name__ == '__main__':
    edges, node_names, layer_names = load_dataset()
    # report()
    # multiplexity_matrix, layer_ids = demo_multiplexity(plot=False)

    # multiplexity_matrix, layer_ids, trials = demo_clustering()
    
    multiplexity_matrix, layer_ids = compute_fao_multiplexity()
    demo_elbow_method(multiplexity_matrix)
