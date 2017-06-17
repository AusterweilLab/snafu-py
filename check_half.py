#!/usr/bin/python

import pickle
import numpy as np
import rw
import networkx as nx
import sys

filename = sys.argv[1]

usf_graph, usf_items = rw.read_csv("./snet/USF_animal_subset.snet")
usf_graph_nx = nx.from_numpy_matrix(usf_graph)
usf_numnodes = len(usf_items)

td=rw.Data({
        'numx': 3,
        'trim': 35})


fh = open(filename,'r')
alldata = pickle.load(fh)
Xs = alldata['datab'][0:10]
graphs = alldata['uinvite_graphs']
items = alldata['items'][0:10]
priordict = alldata['priordict']

# for old pickles. ha, that's kinda funny.
if 'DEFAULTPRIOR' not in priordict.keys():
    priordict['DEFAULTPRIOR'] = 0.5

# recompute for halfa... should be identical for a=1
#priordict = rw.genGraphPrior(graphs, items, a_inc=0.5)

print rw.probXhierarchical(Xs, graphs, items, priordict, td)

for cut in [j/20.0 for j in range(21)]:
    uinvite_group_graph = rw.priorToGraph(priordict, usf_items,cutoff=cut)
    asd=rw.costSDT(uinvite_group_graph, usf_graph)
    print ",".join([str(i) for i in [filename.split('.')[0], cut, asd[0], asd[2], asd[1]+asd[2]]])

# newmixturepe3 >> 281
# newmixturep275 >> 275
# newmixturep4 + cutoff=.8 >> 267

# constructed with a_inc=1
# [109, 284, 33, 12294]
# -8531.6869171978178

# constructed with a_inc=0.5, prob checked with new prior a_inc=1.0
# [141, 252, 31, 12296]
# -8474.0454830370072

# constructed with a_inc=0.5
# [198, 195, 75, 12252]
# -11785.998043645001


# LL of data excluding prior is better with a=1 than a=0.5
# -852.79026649054151 vs -952.25959742129692
# not helpful...


# rw start
#-8652.5339440088173
#[104, 289, 25, 12302]
