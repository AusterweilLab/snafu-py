import rw
import networkx as nx
import numpy as np
import pickle
import sys

usf_graph, usf_items = rw.read_csv("./snet/USF_full.snet")
usf_graph_nx = nx.from_numpy_matrix(usf_graph)
usf_numnodes = len(usf_items)

with open('dedyne_triadic.csv','r') as triadic_data:
    triadic_data.readline()
    for line in triadic_data:
        line = line.rstrip().split(',')
        res = rw.triadicComparison(usf_graph, usf_items, [line[0], line[1], line[2]])
        print res
        

for i in range(100):
    c1=np.random.choice(len(usf_items))
    c2=np.random.choice(len(usf_items))
    c3=np.random.choice(len(usf_items))
    triad = [usf_items[c1], usf_items[c2], usf_items[c3]]
    print triad
    raw_input()
    res = rw.triadicComparison(usf_graph, usf_items, triad)
    print res
