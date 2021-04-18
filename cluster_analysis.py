from types import SimpleNamespace as ns

import numpy as np
import pandas as pd
import toolz
from sklearn.metrics import silhouette_score

from utils import fallback


@fallback
def try_clustering(model, pairwise_similarity, names=None, verbose=True):
    pairwise_distance = pairwise_similarity.max() - pairwise_similarity
    output_ids = model.fit_predict(pairwise_distance)
    
    unk_labels, cluster_sizes = np.unique(output_ids, return_counts=True)
    num_clusters = np.count_nonzero(unk_labels >= 0)
    non_outliers = np.flatnonzero(output_ids >= 0)
    num_outliers = len(output_ids) - len(non_outliers)

    if len(non_outliers):
        score = silhouette_score(
            pairwise_distance[np.ix_(non_outliers, non_outliers)],
            output_ids[non_outliers],
            metric='precomputed'
        )
    else:
        score = -1
    
    result = ns(
        score=score,
        num_clusters=num_clusters,
        labels=output_ids,
        num_outliers=num_outliers,
        unk_labels=unk_labels,
        cluster_sizes=cluster_sizes
    )
    
    if verbose:
        show_clusters(result, names)        

    return result


def show_clusters(trial_output, names):
    if trial_output.num_clusters == 0:
        print("No clusters")
    
    names = list(names)
    for clust_id in range(trial_output.num_clusters):
        layer_ids, *_ = np.where(trial_output.labels == clust_id)
        layer_names = toolz.get(layer_ids.tolist(), names) 
        print('CLUSTER', clust_id, layer_names)  
    
    if trial_output.num_outliers:
        print("OUTLIERS NUM:", trial_output.num_outliers)


def demo_elbow_method(multiplexity_matrix):
    """
    Performs agglomarative clustering with different cut-off levels,
    display silhouette scores.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster import hierarchy
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import squareform

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
    
    src_dist_matrix = multiplexity_matrix.max() - multiplexity_matrix
    num_objects = len(src_dist_matrix)
    
    # Converting (possibly) asymmetric distance matrix to symmetric
    # pairwise distance array as expected by scipy clustering
    pdist_array = [
        (src_dist_matrix[i, j] + src_dist_matrix[j, i]) / 2
        for i in range(num_objects)
        for j in range(i + 1, num_objects)
    ]
    pdist_matrix = squareform(pdist_array)

    linkage = hierarchy.average(pdist_array)
    nn = np.arange(2, 31)
    scores = []
    for n in nn:
        labels = hierarchy.fcluster(linkage, n, criterion='maxclust')
        scores.append(silhouette_score(
            pdist_matrix,
            labels,
            metric='precomputed'
        ))
    scores = pd.DataFrame({'Number of clusters': nn, 'Silhouette score': scores})

    plt.title('Agglomerative clustering')
    sns.lineplot(
        data=scores,
        x='Number of clusters',
        y='Silhouette score',
        markers=False,
        dashes=True
    )
