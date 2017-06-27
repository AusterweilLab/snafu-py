import rw
import networkx as nx
import numpy as np
import pickle
import sys

usf_graph, usf_items = rw.read_csv("./snet/USF_animal_subset.snet")
usf_graph_nx = nx.from_numpy_matrix(usf_graph)
usf_numnodes = len(usf_items)

numlists = 30
listlength = 35
methods=['uinvite']

td=rw.Data({
        'numx': numlists,
        'trim': listlength })

fitinfo=rw.Fitinfo({
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf })

irts=rw.Irts({
        'data': [],
        'irttype': "gamma",
        'irt_weight': 0.95,
        'rcutoff': 20})

# make sure we cover the whole graph (for testing... it's easier this way)
numitems=0
while numitems < usf_numnodes:
    Xs, irts.data = rw.genX(usf_graph_nx, td)[:2]
    numitems = len(set(rw.flatten_list(Xs)))
irts.data = rw.stepsToIRT(irts)

fh=open('irt_output.csv','w')
for irt_weight in [0.5, 0.9, 0.95, 0.7]:
    graph, ll = rw.uinvite(Xs, td, usf_numnodes, irts=irts, fitinfo=fitinfo)
    
    costlist = [rw.costSDT(graph, usf_graph), rw.cost(graph, usf_graph)]
    costlist = rw.flatten_list(costlist)
    costlist = [irt_weight] + costlist
    costlist = [str(i) for i in costlist]
    coststring = ','.join(costlist) + '\n'
    fh.write(coststring)

fh.close()
