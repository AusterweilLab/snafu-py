# First let's generate a network so we can calculate network metrics
# This snippet is copied from estimate_networks.py

import snafu
import numpy as np
import pickle

filepath = "../fluency_data/snafu_sample.csv"
category="animals"

fitinfo = snafu.Fitinfo({
        'cn_alpha': 0.05,
        'cn_windowsize': 2,
        'cn_threshold': 2,
        })

fluencydata = snafu.load_fluency_data(filepath,
                category=category,
                removePerseverations=True,
                spell="../spellfiles/animals_snafu_spellfile.csv",
                hierarchical=False,
                group="Experiment1")

cn_graph = snafu.conceptualNetwork(fluencydata.Xs, numnodes=fluencydata.groupnumnodes, fitinfo=fitinfo)


# NetworkX implements many network measure calculations. First you will have to
# convert your network from a numpy matrix to NetworkX graph

import networkx as nx
nx_graph = nx.Graph(cn_graph)

# Many measures can be computed as one-liners

clustering_coefficient = nx.average_clustering(nx_graph)
density = nx.density(nx_graph)
number_of_edges = nx.number_of_edges(nx_graph)
number_of_nodes = nx.number_of_nodes(nx_graph)

node_degrees = nx.average_neighbor_degree(nx_graph)
average_node_degree = np.mean(list(node_degrees.values()))

# Some measures do not work on networks that are not connected. A common
# work-around is to use only the largest component (and then document how the
# largest component differs from the full network in your manuscript)

nodes_in_largest_component = max(nx.connected_components(nx_graph), key=len)
largest_component = nx_graph.subgraph(nodes_in_largest_component)

average_shortest_path_length = nx.average_shortest_path_length(largest_component)
diameter = nx.diameter(largest_component)

metrics = {
    "clustering_coefficient": clustering_coefficient,
    "density": density,
    "number_of_edges": number_of_edges,
    "number_of_nodes": number_of_nodes,
    "average_node_degree": average_node_degree,
    "average_shortest_path_length": average_shortest_path_length,
    "diameter": diameter
}

with open("demos_data/cn_metrics_expected.pkl", "wb") as f:
    pickle.dump(metrics, f)

# NetworkX can do many many other network calculations. Google for the documentation, which is quite good.
