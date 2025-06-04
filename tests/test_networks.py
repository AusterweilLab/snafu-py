import snafu
import networkx as nx
import numpy as np
import pickle
import pytest

filepath = "../fluency_data/snafu_sample.csv"
category = "animals"

fitinfo = snafu.Fitinfo({
    'cn_alpha': 0.05,
    'cn_windowsize': 2,
    'cn_threshold': 2
})

fluencydata = snafu.load_fluency_data(filepath,
                category=category,
                removePerseverations=True,
                spell="../spellfiles/animals_snafu_spellfile.csv",
                hierarchical=False,
                group="Experiment1")

def test_conceptual_network_metrics():
    cn_graph = snafu.conceptualNetwork(fluencydata.Xs, numnodes=fluencydata.groupnumnodes, fitinfo=fitinfo)
    nx_graph = nx.Graph(cn_graph)

    # Compute metrics
    clustering = nx.average_clustering(nx_graph)
    density = nx.density(nx_graph)
    num_edges = nx.number_of_edges(nx_graph)
    num_nodes = nx.number_of_nodes(nx_graph)
    avg_neighbor_degree = nx.average_neighbor_degree(nx_graph)
    avg_node_degree = np.mean(list(avg_neighbor_degree.values()))

    largest_cc = max(nx.connected_components(nx_graph), key=len)
    subgraph = nx_graph.subgraph(largest_cc)

    avg_path_len = nx.average_shortest_path_length(subgraph)
    diameter = nx.diameter(subgraph)

    with open("../demos_data/cn_metrics_expected.pkl", "rb") as f:
        expected = pickle.load(f)

    def approx_equal(a, b, tol=1e-6):
        return abs(a - b) <= tol

    assert clustering == expected["clustering_coefficient"]
    assert density == expected["density"]

    assert approx_equal(clustering, expected["clustering_coefficient"])
    assert approx_equal(density, expected["density"])
    assert num_edges == expected["number_of_edges"]
    assert num_nodes == expected["number_of_nodes"]
    assert approx_equal(avg_node_degree, expected["average_node_degree"])
    assert approx_equal(avg_path_len, expected["average_shortest_path_length"])
    assert diameter == expected["diameter"]