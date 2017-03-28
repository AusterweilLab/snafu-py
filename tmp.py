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
        'directed': False,
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

subs=['S102','S103','S104','S105','S106','S107','S108','S109','S110',
      'S111','S112','S113','S114','S115','S116','S117','S118','S119','S120']

# u-invite
for subj in subs:
    category="animals"
    Xs, items, irts.data, numnodes=rw.readX(subj,category,'./Spring2015/results_cleaned.csv',ignorePerseverations=True)
    toydata.numx = len(Xs)

    uinvite_graph, bestval=rw.uinvite(Xs, toydata, numnodes, fitinfo=fitinfo)

    #orig=rw.probX(Xs, uinvite_graph, toydata)

    extra_data={}

    for inum, i in enumerate(uinvite_graph):
        for jnum, j in enumerate(i):
            if uinvite_graph[inum,jnum]==1:
                uinvite_graph[inum,jnum]=0
                uinvite_graph[jnum,inum]=0
                result=rw.probX(Xs, uinvite_graph, toydata, forceCompute=True)
                if items[inum] not in extra_data:
                    extra_data[items[inum]]={}
                if items[jnum] not in extra_data:
                    extra_data[items[jnum]]={}
                extra_data[items[inum]][items[jnum]] = (result[0], np.mean(rw.flatten_list(result[1])))
                extra_data[items[jnum]][items[inum]] = (result[0], np.mean(rw.flatten_list(result[1])))
                uinvite_graph[inum,jnum]=1
                uinvite_graph[jnum,inum]=1

    g=nx.to_networkx_graph(uinvite_graph)
    nx.relabel_nodes(g, items, copy=False)
    rw.write_csv(g,subj+".csv",subj,extra_data=extra_data) # write multiple graphs


#@joe
#what if we only include edges that are either bidirectional or uni-directional AND in the undirected graph
#there's also a standard procedure for converting directed graphs into undirected graphs
#it's called moralization

# match edge density of USF network (i.e., prior on edges)
