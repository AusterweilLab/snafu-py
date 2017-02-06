# Geneate graphs from real data

import networkx as nx
import rw
import numpy as np

subs=['S101','S102','S103','S104','S105','S106','S107','S108','S109','S110',
      'S111','S112','S113','S114','S115','S116','S117','S118','S119','S120']

toydata=rw.Toydata({
        'numx': 3,
        'trim': 1,
        'jump': 0.0,
        'jumptype': "stationary",
        'startX': "stationary"})

fitinfo=rw.Fitinfo({
        'startGraph': "windowgraph",
        'windowgraph_size': 2,
        'windowgraph_threshold': 2,
        'followtype': "avg", 
        'prior_samplesize': 10000,
        'recorddir': "records/",
        'prune_limit': 100,
        'triangle_limit': 100,
        'other_limit': 100})

toygraphs=rw.Toygraphs({
        'numgraphs': 1,
        'graphtype': "steyvers",
        'numnodes': 280,
        'numlinks': 6,
        'prob_rewire': .3})

irts=rw.Irts({
        'data': [],
        'irttype': "gamma",
        'beta': (1/1.1), 
        'irt_weight': 0.9,
        'rcutoff': 20})


for subj in subs:
    category="animals"
    Xs, items, irts.data, numnodes=rw.readX(subj,category,'./Spring2015/results_cleaned.csv')
    best_graph, bestval=rw.findBestGraph(Xs, toydata, numnodes, fitinfo=fitinfo)
    rw_graph=rw.noHidden(Xs, numnodes)
    window_graph=rw.windowGraph(Xs, numnodes, td=toydata, valid=1, fitinfo=fitinfo)
    g=nx.to_networkx_graph(best_graph)
    g2=nx.to_networkx_graph(rw_graph)
    g3=nx.to_networkx_graph(window_graph)
    nx.relabel_nodes(g, items, copy=False)
    nx.relabel_nodes(g2, items, copy=False)
    nx.relabel_nodes(g3, items, copy=False)
    rw.write_csv([g, g2, g3],subj+".csv",subj) # write multiple graphs
