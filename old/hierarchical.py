import networkx as nx
import rw
import numpy as np
import pickle


subs=['S101','S102','S103','S104','S105','S106','S107','S108','S109','S110',
      'S111','S112','S113','S114','S115','S116','S117','S118','S119','S120']
#subs=['S1','S2','S3','S4','S5','S7','S8','S9','S10','S11','S12','S13']

toydata=rw.Toydata({
        'numx': 3,
        'trim': 1,
        'jump': 0.0,
        'jumptype': "stationary",
        'priming': 0.0,
        'startX': "stationary"})

fitinfo=rw.Fitinfo({
        'startGraph': "goni_valid",
        'goni_size': 2,
        'goni_threshold': 2,
        'followtype': "avg", 
        'prior_samplesize': 10000,
        'prior_a': 1,
        'prior_b': 1,
        'recorddir': "records/",
        'directed': False,
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
        'exgauss_lambda': 0.721386887,
        'exgauss_sigma': 6.58655566,
        'irt_weight': 0.95,
        'rcutoff': 20})

graphs=[]
items=[]
Xs=[]
numnodes=[]
irts=[]

# generate starting graphs
for subj in subs:
    category="animals"
    ss_Xs, ss_items, ss_irtdata, ss_numnodes=rw.readX(subj,category,'./Spring2015/results_cleaned.csv',ignorePerseverations=True)
    
    ss_irts=rw.Irts({
        'data': [],
        'irttype': "exgauss",
        'exgauss_lambda': 0.721386887,
        'exgauss_sigma': 6.58655566,
        'irt_weight': 0.95,
        'rcutoff': 20})

    ss_irts.data = ss_irtdata
    irts.append(ss_irts)
    items.append(ss_items)
    Xs.append(ss_Xs)
    numnodes.append(ss_numnodes)

graphs, priordict = rw.hierarchicalUinvite(Xs, items, numnodes, toydata, fitinfo=fitinfo, seed=None, irts=irts)

for subnum, subj in enumerate(subs):
    g=nx.to_networkx_graph(graphs[subnum])
    g2=nx.to_networkx_graph(rw.noHidden(Xs[subnum],numnodes[subnum]))
    nx.relabel_nodes(g, items[subnum], copy=False)
    nx.relabel_nodes(g2, items[subnum], copy=False)
    rw.write_csv([g, g2],subj+"_irt.csv",subj) # write multiple graphs
    
with open('prior_irt.pickle','w') as fh:
    pickle.dump(priordict, fh)
