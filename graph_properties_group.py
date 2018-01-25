import rw
import numpy as np
import networkx as nx

filepath = "all.csv"
methods=["chan","fe","goni","kenett","rw","uinvite_flat","uinvite_hierarchical"]

for method in methods:
    graph, items = rw.read_csv(filepath,header=True,cols=("item1","item2"),filters={method: "1"})
    graph = nx.to_networkx_graph(graph)
    num_nodes = nx.number_of_nodes(graph)
    num_edges = nx.number_of_edges(graph)
    num_components = nx.number_connected_components(graph)
    is_connected = nx.is_connected(graph)
    density = nx.density(graph)
    largest_component = max(nx.connected_component_subgraphs(graph), key=len)
    largest_component_size = nx.number_of_nodes(largest_component)  
    aspl = nx.average_shortest_path_length(largest_component)         # !! aspl calculated on largest component
    degree = np.mean(graph.degree().values())
    clustering = nx.average_clustering(graph)
    
    vars_to_print = [method, num_nodes, num_edges, num_components, density, largest_component_size, aspl, clustering, is_connected, degree]
    print ",".join([str(i) for i in vars_to_print])
