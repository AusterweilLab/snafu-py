# compare LL of USF graph (or subset) and naive RW graph, given the data

import numpy as np
import sys
sys.path.append('./rw')
import rw
import math
import random
import networkx as nx
import pygraphviz

usf,items=rw.read_csv('USF_animal_subset.csv')
usfg=nx.to_networkx_graph(usf)
numnodes=len(items)

sdt_rws=[]
sdt_uis=[]

f=open('usf_sims.csv','a', 0)                # write/append to file with no buffering

for gnum in range(100):
    # generate toy lists
    numlists=11
    listlength=160   # must be <= numnodes
    Xs=[rw.genX(usfg)[0:listlength] for i in range(numlists)]

    itemsinXs=np.unique(rw.flatten_list(Xs))    # list of items that appear in toy data

    # reconstruct graph
    rw_graph=rw.noHidden(Xs, numnodes)

    # remove nodes not in X from all graphs before comparison
    rw_graph=rw_graph[itemsinXs,]
    rw_graph=rw_graph[:,itemsinXs]
    usfcopy=np.copy(usf)
    usfcopy=usfcopy[itemsinXs,]
    usfcopy=usfcopy[:,itemsinXs]

    for xnum, x in enumerate(Xs):
        for itemnum, item in enumerate(x):
            Xs[xnum][itemnum] = list(itemsinXs).index(item)

    print "RW: ", rw.probX(Xs,rw_graph,len(itemsinXs)), " USF: ", rw.probX(Xs,usfcopy,len(itemsinXs))
