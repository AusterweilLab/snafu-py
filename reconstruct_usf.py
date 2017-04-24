import rw
import networkx as nx
import numpy as np

usf_graph,usf_items = rw.read_csv("./snet/USF_animal_subset.snet")
usf_graph = nx.from_numpy_matrix(usf_graph)
numnodes = len(usf_items)

numsubs = 50
numlists = 3
listlength = 35

toydata=rw.Data({
        'numx': numlists,
        'trim': listlength })

fitinfo=rw.Fitinfo({
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf})

#toygraphs=rw.Graphs({
#        'numgraphs': 1,
#        'graphtype': "steyvers",
#        'numnodes': 280,
#        'numlinks': 6,
#        'prob_rewire': .3})

#irts=rw.Irts({})

# generate data for `numsub` participants, each having `numlists` lists of `listlengths` items
data = []
for sub in range(numsubs):
    Xs = rw.genX(usf_graph, toydata, seed=None)
    data.append(Xs[0])

uinvite_graph, bestval=rw.uinvite(data[0], toydata, numnodes, fitinfo=fitinfo)
rw_graph=rw.noHidden(Xs, numnodes)
goni_graph=rw.goni(Xs, numnodes, td=toydata, valid=0, fitinfo=fitinfo)


