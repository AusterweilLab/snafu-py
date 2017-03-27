# Geneate graphs from real data

import networkx as nx
import rw
import numpy as np

toydata=rw.Toydata({
        'numx': 3,
        'trim': 1,
        'jump': 0.0,
        'jumptype': "stationary",
        'priming': 0.0,
        'startX': "stationary"})

fitinfo=rw.Fitinfo({
        'startGraph': "windowgraph_valid",
        'windowgraph_size': 2,
        'windowgraph_threshold': 2,
        'followtype': "avg", 
        'undirected': False,
        'prior_samplesize': 10000,
        'recorddir': "records/",
        'prune_limit': np.inf,
        'triangle_limit': np.inf,
        'other_limit': np.inf})

toygraphs=rw.Toygraphs({
        'numgraphs': 1,
        'graphtype': "steyvers",
        'numnodes': 280,
        'numlinks': 6,
        'prob_rewire': .3})

irts=rw.Irts({
        'data': [],
        'irttype': "exgauss",
        'lambda': 0.721386887,
        'sigma': 6.58655566,
        'irt_weight': 0.5,
        'rcutoff': 20})

subj="S101"
category="animals"
Xs, items, irts.data, numnodes=rw.readX(subj,category,'./Spring2015/results_cleaned.csv',ignorePerseverations=True)
toydata.numx = len(Xs)

# u-invite
uinvite_graph, bestval=rw.uinvite(Xs, toydata, numnodes, fitinfo=fitinfo)

orig=rw.probX(Xs, uinvite_graph, toydata)

for inum, i in enumerate(uinvite_graph):
    for jnum, j in enumerate(i):
        if uinvite_graph[inum,jnum]==1:
            uinvite_graph[inum,jnum]=0
            uinvite_graph[jnum,inum]=0
            print rw.probX(Xs, uinvite_graph, toydata)[0]
            uinvite_graph[inum,jnum]=1
            uinvite_graph[jnum,inum]=1


# most important edges -- if removal of edge creates impossible transition
# quantify importance of edges -- mean difference in transition probabilities
# test uinvite+irt+priming+prior
# directed network?
