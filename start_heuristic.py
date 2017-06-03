import pickle
import numpy as np
import rw
import networkx as nx

usf_graph, usf_items = rw.read_csv("./snet/USF_animal_subset.snet")
usf_graph_nx = nx.from_numpy_matrix(usf_graph)
usf_numnodes = len(usf_items)

numsubs = 10
numlists = 3
listlength = 35
numsims = 50
#methods=['rw','goni','chan','kenett','fe']
methods=['uinvite']


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
        'other_limit': np.inf })


seednum =0

data=[]
datab=[]
numnodes=[]
items=[]

for sub in range(numsubs):
    Xs = rw.genX(usf_graph_nx, toydata, seed=seednum)[0]
    data.append(Xs)
                                                         
    # renumber dictionary and item listuinvite_group_graph = rw.priorToGraph(priordict, usf_items)
    itemset = set(rw.flatten_list(Xs))
    numnodes.append(len(itemset))
                                                         
    ss_items = {}
    convertX = {}
    for itemnum, item in enumerate(itemset):
        ss_items[itemnum] = usf_items[item]
        convertX[item] = itemnum
                                                         
    items.append(ss_items)
                                                         
    Xs = [[convertX[i] for i in x] for x in Xs]
    datab.append(Xs)
    
    seednum += numlists

for sub in range(len(Xs)):
    td.numx=len(Xs[sub])    # for goni
    graphs.append(genStartGraph(Xs[sub], numnodes[sub], td, fitinfo=fitinfo))









#priordict = rw.genGraphPrior(graphs, items, a_inc=1.0)
#rw.probXhierarchical(Xs, graphs, items, priordict, td)
#rw.costSDT(uinvite_group_graph, usf_graph)
