import rw
import numpy as np
import networkx as nx

subs=["S"+str(i) for i in range(101,151)]
filepath = "individual_graphs.csv"
methods=["chan","fe","goni","kenett","rw","uinvite_flat","uinvite_hierarchical"]

def property_stats(arr):
    return [str(i) for i in [np.mean(arr), np.std(arr), np.min(arr), np.max(arr)]]

for method in methods:
    aspl = []
    num_nodes = []
    num_edges = []
    num_components = []
    is_connected = []
    largest_component_size = []
    density = []
    clustering = []
    degree = []
    
    for subj in subs:
        graph, items = rw.read_csv(filepath,header=True,cols=("item1","item2"),filters={"method": method, "edge": "1", "subj": subj})
        graph = nx.to_networkx_graph(graph)
        if nx.number_of_edges(graph) > 0:                   # all metrics calculated on non-empty graphs
            num_nodes.append(nx.number_of_nodes(graph))
            num_edges.append(nx.number_of_edges(graph))
            num_components.append(nx.number_connected_components(graph))
            is_connected.append(nx.is_connected(graph))
            density.append(nx.density(graph))
            largest_component = max(nx.connected_component_subgraphs(graph), key=len)
            largest_component_size.append(nx.number_of_nodes(largest_component))
            aspl.append(nx.average_shortest_path_length(largest_component))         # aspl calculated on largest component
            degree.append(np.mean(graph.degree().values()))
            clustering.append(nx.average_clustering(graph))
    
    number_of_graphs = str(len(num_nodes) / float(len(subs)))
    vars_to_print = [method, number_of_graphs] + property_stats(num_nodes) + property_stats(num_edges) + property_stats(num_components) + property_stats(density) + property_stats(largest_component_size) + property_stats(aspl) + property_stats(clustering) + [str(np.mean(is_connected))] + property_stats(degree)
    print ",".join(vars_to_print)
