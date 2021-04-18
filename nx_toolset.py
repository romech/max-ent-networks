import networkx as nx


def edges_to_nx(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    if 'weight' in edges.columns:
        G.add_weighted_edges_from(edges[['node_1', 'node_2', 'weight']].values)
    else:
        G.add_edges_from(edges[['node_1', 'node_2']].values)
    return G


def edges_to_nx_ensemble(layers, nodes, edges):
    GG = {layer_id: edges_to_nx(nodes, edges[edges.layer_id == layer_id])
          for layer_id in layers}
    return GG


if __name__ == '__main__':
    from fao import load_dataset
    
    edges, node_names, layer_names = load_dataset()
    edges = edges[edges.layer_id == 1]
    G = edges_to_nx(node_names.index, edges)
